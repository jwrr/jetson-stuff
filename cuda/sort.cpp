
#include <iostream>

using namespace std;


// ============================================================================
// ============================================================================


void sort(int *a, int n, int stride)
{
  if(stride > 1) sort(a, n, stride / 2);
  for(int i=0; i < n-stride; i++){
    if(a[i] > a[i+stride]){
      swap(a[i], a[i+stride]);
    }
  }
  if(stride > 1) sort(a, n, stride / 2);
}

// assumes n is a power of 2
void sort_array(int *a, int n)
{
  sort(a, n, n / 2);
}


// ============================================================================
// ============================================================================

void print_array(string name, int *a, int n)
{
  cout << name << " = {";
  for(int i = 0; i < n; i++){
    cout << a[i];
    if(i < n-1) cout << ", ";
  }
  cout << "}" << endl;
}

// create array of random numbers
void rand_array(int *a, int n, int max)
{
  for(int i = 0; i < n; i++){
    a[i] = rand() % max;
  }
}

// verify array is monotonically increasing
int test_array(int *a, int n)
{
  int err = 0;
  for(int i = 1; i < n; i++){
    if(a[i] < a[i-1]) return 1;
  }
  return 0;
}


// ============================================================================
// ============================================================================


int main()
{
  int arr_size = 2048;
  int arr[arr_size];

  int errcnt = 0;
  for (int loopcnt=0; loopcnt < 1000; loopcnt++) {
    rand_array(arr, arr_size, 4*arr_size);
    sort_array(arr, arr_size);
    if (int fail = test_array(arr, arr_size)){
      print_array("fail ", arr, arr_size);
      errcnt++;
    }
  } // test loop

  if(errcnt){
    cout << "Test FAILED" << endl;
  } else {
    cout << "Test PASSED" << endl;
  }
} // main

