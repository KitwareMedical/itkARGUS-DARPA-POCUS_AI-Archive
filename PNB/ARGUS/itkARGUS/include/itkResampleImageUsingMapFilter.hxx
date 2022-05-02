/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkResampleImageUsingMapFilter_hxx
#define itkResampleImageUsingMapFilter_hxx

#include "itkResampleImageUsingMapFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkConstantBoundaryCondition.h"

namespace itk
{

template <typename TInputImage, typename TOutputImage>
ResampleImageUsingMapFilter<TInputImage, TOutputImage>
::ResampleImageUsingMapFilter()
{
  m_Interpolate = true;
}


template <typename TInputImage, typename TOutputImage>
void
ResampleImageUsingMapFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream & os, Indent indent) const
{
  os << indent << "OutputSize = " << m_OutputSize << std::endl;
  os << indent << "SourceMapping (size) = " << m_SourceMapping.size()
    << std::endl;
  os << indent << "Kernels (size) = " << m_Kernels.size()
    << std::endl;
  Superclass::PrintSelf(os, indent);
}

template <typename TInputImage, typename TOutputImage>
void
ResampleImageUsingMapFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  // do not call the superclass' implementation of this method since
  // this filter allows the input and the output to be of different dimensions

  // get pointers to the input and output
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput();
  typename Superclass::InputImageConstPointer inputPtr = this->GetInput();

  if (!outputPtr || !inputPtr)
  {
    std::cerr << "ERROR: input or output image not available to filter"
      << std::endl;
    return;
  }

  auto area = m_OutputSize[0] * m_OutputSize[1];
  if (m_SourceMapping.size() != 2 * area) {
    itkExceptionMacro(<< "itk::ResampleImageUsingMapFilter::GenerateOutputInformation "
                      << "mismatch between source mapping size and output size");
  }
  if (m_Kernels.size() != 9 * area) {
    itkExceptionMacro(<< "itk::ResampleImageUsingMapFilter::GenerateOutputInformation "
                      << "mismatch between kernels vector size and output size");
  }

  typename InputImageType::RegionType outputRegion;
  typename InputImageType::IndexType origin = {0, 0};
  outputRegion.SetSize(m_OutputSize);
  outputRegion.SetIndex(origin);

  // Set the output image size to the same value as the extraction region.
  outputPtr->SetLargestPossibleRegion(outputRegion);
  outputPtr->SetSpacing(inputPtr->GetSpacing());
  outputPtr->SetDirection(inputPtr->GetDirection());
  outputPtr->SetOrigin(inputPtr->GetOrigin());
  outputPtr->SetNumberOfComponentsPerPixel(inputPtr->GetNumberOfComponentsPerPixel());
}

template <typename TInputImage, typename TOutputImage>
void
ResampleImageUsingMapFilter<TInputImage, TOutputImage>
::DynamicThreadedGenerateData(const OutputRegionType & outputRegion)
{
  OutputImageType *      output = this->GetOutput();
  const InputImageType * input = this->GetInput();
  using InputRegionType = typename InputImageType::RegionType;
  using InputSizeType = typename InputImageType::SizeType;
  using InputIndexType = typename InputImageType::IndexType;
  using InputOffsetType = typename InputImageType::OffsetType;

  using BoundaryConditionType = itk::ConstantBoundaryCondition<InputImageType, OutputImageType>;
  BoundaryConditionType boundedAccessor;
  boundedAccessor.SetConstant(0);

  itk::ImageRegionIterator<OutputImageType>     out(output, outputRegion);

  /* treat kernels as row-major
  InputOffsetType kernelIndexOffsets[] = {{-1, -1}, {-1, 0}, {-1, 1},
                                          { 0, -1}, { 0, 0}, { 0, 1},
                                          { 1, -1}, { 1, 0}, { 1, 1}};
                                          */

  // kernels are column-major
  InputOffsetType kernelIndexOffsets[] = {{-1, -1}, { 0,-1}, { 1,-1},
                                          {-1,  0}, { 0, 0}, { 1, 0},
                                          {-1,  1}, { 0, 1}, { 1, 1}};

  InputIndexType inputIndex = {0,0};
  for (out.GoToBegin(); !out.IsAtEnd(); ++out)
  {
    auto index = out.GetIndex();
    auto offset = index[1] * m_OutputSize[0] + index[0]; // row-major

    auto sourceMapOffset = 2 * offset; // 2 elements per source mapping
    inputIndex[1] = m_SourceMapping[sourceMapOffset];
    inputIndex[0] = m_SourceMapping[sourceMapOffset+1];
    if( inputIndex[0]<=1 || inputIndex[1]<=1 )
    {
      out.Set(0);
      continue;
    }

    OutputPixelType pixel = 0;
    if( m_Interpolate )
    {
      auto kernelOffset = 9 * offset; // 9 elements per kernel
      for (int i = 0; i < 9; i++)
      {
        auto kernel_i = m_Kernels[kernelOffset + i];
        auto indexOffset = kernelIndexOffsets[i];
        pixel += kernel_i * boundedAccessor.GetPixel(inputIndex + indexOffset, input);
      }
    }
    else
    {
      pixel = boundedAccessor.GetPixel(inputIndex, input);
      if (pixel != 0)
      {
        auto kernelOffset = 9 * offset; // 9 elements per kernel
        double matchWeight = 0;
        double mismatchWeight = 0;
        for (int i = 0; i < 9; i++)
        {
          auto kernel_i = m_Kernels[kernelOffset + i];
          auto indexOffset = kernelIndexOffsets[i];
          if (pixel == boundedAccessor.GetPixel(inputIndex+indexOffset, input))
          {
            matchWeight += kernel_i;
          }
          else
          {
            mismatchWeight += kernel_i;
          }
        }
        if (mismatchWeight > matchWeight)
        {
          pixel = 0;
        }
      }
    }

    out.Set(pixel);
  }
}

} // end namespace itk

#endif // itkResampleImageUsingMapFilter_hxx
