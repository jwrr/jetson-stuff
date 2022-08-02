
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
  
  cv::Mat image1 = image0;
  
  int white = 256;
  int width = center_col / 2;
  int height = 2 * center_row;
  cv::Point upper_left(center_col - width/2, center_row - height/2);
  cv::Point lower_right(center_col + width/2, center_row + height/2);
  cv::rectangle(image1, upper_left, lower_right, white, thickness);
  
  cv::imshow("Circle + Rectangle", image0);
  cv::waitKey();
  return 0;
}



