#ifndef  __UTILS_HPP__
#define __UTILS_HPP__
#include "Blob.hpp"
//#include <string>

#include <memory>
using std::shared_ptr;

void ReadMnistData(string path, shared_ptr<Blob> &images);
void ReadMnistLabel(string path, shared_ptr<Blob> &labels);

#endif  //__UTILS_HPP__