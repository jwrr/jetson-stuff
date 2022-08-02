
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
  cv::Mat image0;
  
  int rows = 480;
  int cols = 640;
  int format = CV_8UC1; // 8-bit mono
  
  // Note: order is rows, cols
  image0.create(rows, cols,  format);
  image0.setTo(0);
  
  int center_col = image0.cols / 2;
  int center_row = image0.rows / 2;
  
  // Note: order is col, row
  cv::Point center(center_col, center_row);
  
  int color = 128;
  int thickness = 3;
  int radius = center_row - thickness;
  
  cv::circle(image0, center, radius, color, thickness);
  cv::imshow("Circle", image0);
  cv::waitKey();
  return 0;
}



