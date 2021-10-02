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
#ifndef itkResampleImageUsingMapFilter_h
#define itkResampleImageUsingMapFilter_h

#include "itkImageToImageFilter.h"

namespace itk
{

/** \class ResampleImageUsingMapFilter
 *
 * \brief Filters a image by iterating over its pixels.
 *
 * Filters a image by iterating over its pixels in a multi-threaded way
 * and {to be completed by the developer}.
 *
 * \ingroup ResampleImageUsingMapFilter
 * \ingroup itkARGUS
 *
 */
template <typename TInputImage, typename TOutputImage>
class ResampleImageUsingMapFilter : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ResampleImageUsingMapFilter);

  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;
  static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;

  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using InputPixelType = typename InputImageType::PixelType;
  using OutputPixelType = typename OutputImageType::PixelType;
  using OutputSizeType = typename OutputImageType::SizeType;

  /** Standard class aliases. */
  using Self = ResampleImageUsingMapFilter<InputImageType, OutputImageType>;
  using Superclass = ImageToImageFilter<InputImageType, OutputImageType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information. */
  itkTypeMacro(ResampleImageUsingMapFilter, ImageToImageFilter);

  /** output image size **/
  itkSetMacro(OutputSize, OutputSizeType);
  itkGetMacro(OutputSize, OutputSizeType);

  /** source mappings
   * vector should be row-major, but points should be in Y,X order
   */
  virtual void SetSourceMapping(std::vector<int> mapping) {
    this->m_SourceMapping = mapping;
  }
  itkGetMacro(SourceMapping, std::vector<int>);

  /** point kernels
   * expects 3x3 kernels for every point.
   * Kernels are column-major, and vector is row-major.
   */
  virtual void SetKernels(std::vector<float> kernels) {
    this->m_Kernels = kernels;
  }
  itkGetMacro(Kernels, std::vector<float>);

  itkSetMacro(Interpolate, bool);
  itkGetMacro(Interpolate, bool);

  /** Standard New macro. */
  itkNewMacro(Self);

protected:
  ResampleImageUsingMapFilter();
  ~ResampleImageUsingMapFilter() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  using OutputRegionType = typename OutputImageType::RegionType;
  /** ResampleImageUsingMap specifies output size.
   * \sa ProcessObject::GenerateOutputInformation() */
  void
  GenerateOutputInformation() override;


  void
  DynamicThreadedGenerateData(const OutputRegionType & outputRegion) override;

private:
  OutputSizeType m_OutputSize;
  std::vector<int> m_SourceMapping;
  std::vector<float> m_Kernels;
  bool m_Interpolate;
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkResampleImageUsingMapFilter.hxx"
#endif

#endif // itkResampleImageUsingMapFilter
