itk_module_test()

set(itkARGUSTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkResampleImageUsingMapFilterTest.cxx
  )

CreateTestDriver(itkARGUS "${itkARGUS-Test_LIBRARIES}" "${itkARGUSTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND itkARGUSTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

itk_add_test(NAME itkResampleImageUsingMapFilterTest
  COMMAND itkARGUSTestDriver
    --compare
    DATA{Baseline/itkResampleImageUsingMapFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkResampleImageUsingMapFilterTestOutput.mha
  itkResampleImageUsingMapFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkResampleImageUsingMapFilterTestOutput.mha
  )
