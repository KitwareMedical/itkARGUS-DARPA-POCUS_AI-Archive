itk_module_test()

set(CurvilinearResampleFilterTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkCurvilinearResampleFilterTest.cxx
  )

CreateTestDriver(CurvilinearResampleFilter "${CurvilinearResampleFilter-Test_LIBRARIES}" "${CurvilinearResampleFilterTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND CurvilinearResampleFilterTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

itk_add_test(NAME itkCurvilinearResampleFilterTest
  COMMAND CurvilinearResampleFilterTestDriver
    --compare
    DATA{Baseline/itkCurvilinearResampleFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkCurvilinearResampleFilterTestOutput.mha
  itkCurvilinearResampleFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkCurvilinearResampleFilterTestOutput.mha
  )
