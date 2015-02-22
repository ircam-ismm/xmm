//
//  main.cpp
//  xmm-tester
//
//  Created by Jules Francoise on 22/02/2015.
//
//

#include <iostream>
#include "xmm.h"

int main(int argc, const char * argv[]) {
    GMM *gmm_test = new GMM();
    gmm_test->train();
    std::cout << "Hello, World!\n";
    return 0;
}
