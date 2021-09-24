itk_module_test()

set(itkARGUSTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkCurvilinearResampleFilterTest.cxx
  )

CreateTestDriver(itkARGUS "${itkARGUS-Test_LIBRARIES}" "${itkARGUSTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND itkARGUSTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

itk_add_test(NAME itkCurvilinearResampleFilterTest
  COMMAND itkARGUSTestDriver
    --compare
    DATA{Baseline/itkCurvilinearResampleFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkCurvilinearResampleFilterTestOutput.mha
  itkCurvilinearResampleFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkCurvilinearResampleFilterTestOutput.mha
  )
