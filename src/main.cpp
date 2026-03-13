#include <iostream>
#include <Eigen/Dense>

int main() {
  Eigen::Matrix2d m = Eigen::Matrix2d::Random();
  std::cout << "Eigen is working! Here is a random matrix:\n" << m << std::endl;
  return 0;
}
