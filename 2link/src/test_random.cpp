#include<iostream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

using namespace std;

static boost::random::mt19937 gen;

float get_random(){
  static boost::random::uniform_real_distribution<> dist(0, 1);
  return dist(gen);
}

float get_rand(){
  return ((float) rand()) / (float) RAND_MAX;
}

int main(){
  // srand(time(NULL));
  gen.seed(time(0));
  cout << get_random() << endl;
  cout << get_random() << endl;
  return 0;
}
