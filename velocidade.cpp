/*
This program is used to detect motorcycles on a designated area of the screen.
Click four times to designate the area of the screen to detect. Start with the
upper left and go clockwise. Then, drag the red line to indicate a prohibited 
zone to the left, and drag the blue line with the mouse to indicate a prohibited area to the right. Hit esc when finished after each drawing. Then drag lines to indicate lanes, if any. The program with run
automatically after pressing escape.

Once running, press q or esc to quit, or space to pause/unpause.

PARAMETERS / USAGE: all parameters optional. Defaults to not drawing lanes and prohibited areas.

./velocidade [0 = don't draw left, any other value draws] [draw right] [number of lanes to draw] [lane detection width in pixels] [distance of area drawn for velocity measurement]

*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>
#include <vector>
#include <ctime>

const int KEY_SPACE = 32;
const int KEY_ESC = 27;

//Size that image will be warped to.
const int X_SIZE = 400;
const int Y_SIZE = 400;

// parameters for motorcycle detection. MAX_WIDTH prevents larger objects being detected.
// MIN_HEIGHT specifies minimum height for motorcycle to be registered.
// In the future, can also be used for car detection. Cars must be GREATER than max_width, and greater than min_height
const int MAX_WIDTH = X_SIZE / 8;
const int MAX_HEIGHT = Y_SIZE / 3;
const int MIN_HEIGHT = Y_SIZE / 6;
const int MIN_WIDTH = X_SIZE / 13;

// Perspective transformation points
cv::Point2f source_points[] = {cv::Point2f(0,0),cv::Point2f(0,0),cv::Point2f(0,0),cv::Point2f(0,0)};
cv::Point2f dst_points[] = {cv::Point2f(0,0), cv::Point2f(X_SIZE,0),cv::Point2f(X_SIZE,Y_SIZE),cv::Point2f(0,Y_SIZE)};

//for shoulder detection. Hold values for limits of prohibited/interest zones
int leftX = 0;
int rightX = 0;

//for in between lane detection
std::vector <int> lanes;
int lanenumber = 0;

// for use in transformation drawing callback.
int pointnumber = 0;

// for drawing prohibited areas.
bool clicked = false;

struct vehicle
{
  //can infer last x and last y during loop, thus do not need to save.
  int initial_y = 0;
  int initial_x = 0;
  int second_x = 0;
  bool type = false; //default not motorcycle
  bool over_line = true; //default to already over the line
};

//Method prototypes
void Erosion(cv::Mat &img,int radius);
void Detect(cv::Mat &img, std::vector<cv::Rect> &objects, std::vector<vehicle> &vehicles);
void Dilation(cv::Mat &img, int radius);
void onMouse(int event, int x, int y, int f, void*);
void leftShoulder(int event, int x, int y, int f, void*);
void rightShoulder(int event, int x, int y, int f, void*);
void drawLane(int event, int x, int y, int f, void*);

int main(int argc, char **argv)
{
  std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;
  
  //Parameters that don't need to be global ----------------------------------------------------------------------------

  //Threshold value for detecting movement blobs. 255 - most difference, 0 - no difference.
  //make higher to reduce noise and prevent overlap, make lower to ensure vehicles are each one blob.
  //30 for cam 71
  const int MIN_THRESH = 27;

  //Rate at which program prints information on count of vehicles and average velocities. 
  const int SAVE_RATE = 60000;

  //Minimum velocity of car. Use to reduce recording velocity of false positives.
  const int MIN_VELOCITY = 6;

  //Detects cars every __ frames. On other frames, tracks.
  const int DETECT_RATE = 4;

  //for mean shift algorithm for detection.
  const int MAX_ITER = 2;

  //epsilon for mean shift. mean numbers of pixels must move each time before stopping.
  const int MS_EPSILON = 1;

  //number of images to initialiaze background subtractor. Used to make median image.
  const int INIT_IMAGES = 300;

  //max distance from lane lines to be counted to inter-lane vehicles.
  int LANE_THRESH = 10;

  //number of lanes to draw
  int number_lanes = 0;

  //for background subtraction. College INIT_IMAGES frames, then get the median value of each pixel
  unsigned char (*image_values) = new unsigned char [INIT_IMAGES*Y_SIZE*X_SIZE];
  
  //for measuring time to calculate motorcycles per hour
  clock_t start_outer, start_inner;
  double duration =.01;

  // for measuring velocities. lenth_kmeters is the length in kilometers of the detection area.
  double length_kmeters = 1.0d;

  std::vector<vehicle> vehicles;
  std::vector <cv::Rect> objects;

  //hold velocities of vehicles. Get cleared every SAVE_RATE frames.
  std::vector<double> moto_velocities;
  std::vector<double> car_velocities;

  //loop counter variable
  unsigned long i = 0;

  //input key
  char key = '\0';

  //number of vehicles/motorcycles that have passed.
  int count = 0;
  
  //left and right prohibited areas
  int leftcount = 0;
  int rightcount = 0;
  int interlanecount = 0;
  int zigzagcount = 0;

  int drawleft = 1;
  int drawright = 1;

  if(argc >= 2)
    {
      drawleft = atoi(argv[1]); 
    }

  if(argc >= 3)
    {
      drawright = atoi(argv[2]);
    }

  if(argc >= 4)
    {
      number_lanes = atoi(argv[3]);
    }
  
  if(argc >= 5)
    {
      LANE_THRESH = atoi(argv[4]);

    }
  if(argc >= 6)
    {
      length_kmeters = atof(argv[5])/1000.0d;
    }

  // end parameters ---------------------------------------------------------------------------------------------------
    
  cv::Mat frame, fgMask, dst;

  //used for mean shift
  cv::TermCriteria criteria(MAX_ITER,MAX_ITER,MS_EPSILON);

  //cv::VideoCapture cap("https://d1h84if288zv9w.cloudfront.net/7dias/0be3_477.stream/chunklist.m3u8");
  cv::VideoCapture cap("Video_1.avi");
  
  if(!cap.isOpened())
    {
      std::cout << "video error " << std::endl;
      return -1;
    }

  cv::namedWindow("Perspective transform",1);
  
  cap >> frame;

  // Capturing user input to create perspective transform --------------------
  cv::imshow("Perspective transform", frame);
  cv::setMouseCallback("Perspective transform",onMouse,NULL);
  
  while(pointnumber < 4 )
    {
      char c = cv::waitKey(1);
      if (c == KEY_ESC) break;
    }
  // Done capturing user input for perspective transform ---------------------

  cv::Mat transform_matrix = cv::getPerspectiveTransform(source_points,dst_points);
  cv::Mat inverse_matrix = cv::getPerspectiveTransform(dst_points,source_points);
  //std::cout << transform_matrix << std::endl;
  cv::warpPerspective(frame,dst,transform_matrix,cv::Size(X_SIZE,Y_SIZE));
  cv::imshow("Perspective transform",dst);

  // Creating prohibited areas -----------------------------------------------
  if (drawleft)
    {
      clicked = false;
      cv::setMouseCallback("Perspective transform",leftShoulder,NULL);
      
      for(;;)
	{
	  frame = dst.clone();
	  cv::line(frame,cv::Point(leftX,0),cv::Point(leftX,dst.rows-1),cv::Scalar(0,0,240),2);
	  cv::imshow("Perspective transform", frame);
	  char c = cv::waitKey(1);
	  if (c == 27) break;
	}
      
      cv::line(dst,cv::Point(leftX,0),cv::Point(leftX,dst.rows-1),cv::Scalar(0,0,240),2);
      clicked = false;
      cv::setMouseCallback("Perspective transform",rightShoulder,NULL);
    }
  
  if(drawright)
    {
      for(;;)
	{
	  frame = dst.clone();
	  cv::line(frame,cv::Point(rightX,0),cv::Point(rightX,dst.rows-1),cv::Scalar(255,0,0),2);
	  cv::imshow("Perspective transform", frame);
	  char c = cv::waitKey(1);
	  if (c == 27) break;
	}
      cv::line(dst,cv::Point(rightX,0),cv::Point(rightX,dst.rows-1),cv::Scalar(255,0,0),2);
    }
  else
    {
      rightX = dst.cols -1;
    }
  // Done creating  prohibited areas ---------------------------------------

  // drawing lanes 
  for (lanenumber = 0; lanenumber< number_lanes; lanenumber++)
    {
      clicked = false;
      lanes.push_back(0);
      cv::setMouseCallback("Perspective transform",drawLane,NULL);
      for(;;)
	{
	  frame = dst.clone();
	  cv::line(frame,cv::Point(lanes.at(lanenumber),0),cv::Point(lanes.at(lanenumber),dst.rows-1),cv::Scalar(0,255,0),2);
	  cv::imshow("Perspective transform", frame);
	  char c = cv::waitKey(1);
	  if (c == 27) break;
	}
      cv::line(dst,cv::Point(lanes.at(lanenumber),0),cv::Point(lanes.at(lanenumber),dst.rows-1),cv::Scalar(0,255,0),LANE_THRESH);
    }

  //prevent more mouse input from causing errors
  cv::setMouseCallback("Perspective transform",NULL,NULL);

  cv::imshow("Perspective transform", dst);

  //if no distance supplied as command line argument, require user to input distance.
  if (argc < 5)
    {
      std::string distance_input = "";
      std::cout << "Input the distance of the vertical stretch in meters for velocity measuring: " << std::endl;
      std::cin >> distance_input;
      length_kmeters = atof(distance_input.c_str())/1000.0d;
      if (length_kmeters <= 0)
	{
	  length_kmeters = .01;
	}
    }

  //std::cout << transform_matrix << std::endl;

  cv::namedWindow("Frame", 1);
  cv::namedWindow("Mask", 1);

  // Main loop ----------------------------------------------------------------------
  while(true)
    {
      count = 0;
      leftcount = 0;
      rightcount = 0;
      interlanecount = 0;
      i = 0;

      //allow background to initialize before starting detection ------------------------
      
      cap >> frame;
      cv::warpPerspective(frame,dst,transform_matrix,cv::Size(X_SIZE,Y_SIZE));
      cv::Mat average;
      cv::cvtColor(dst,dst,CV_BGR2GRAY);
      average = dst.clone();
      average.setTo(cv::Scalar(0,0,0));
      
      for (i=0; i<INIT_IMAGES; i++)
	{
	  cap >> dst;
	  
	  if (dst.empty() || dst.rows == 0 || dst.cols == 0)
	    break;
	  
	  cv::cvtColor(dst,dst,CV_BGR2GRAY);
	  cv::warpPerspective(dst,frame,transform_matrix,cv::Size(X_SIZE,Y_SIZE));
	  for (int y = 0; y < Y_SIZE; y ++)
	    {
	      for (int x = 0; x < X_SIZE; x++)
		{
		  image_values[X_SIZE * INIT_IMAGES * y + INIT_IMAGES * x + i] = frame.at<uchar>(x,y);
		}
	    }
	}
      
      //now sorting images
      for (int y = 0; y < Y_SIZE; y++)
	{
	  for (int x = 0; x < X_SIZE; x++)
	    {
	      int offset = X_SIZE * INIT_IMAGES * y + INIT_IMAGES * x;
	      std::sort(image_values + offset, image_values + offset +INIT_IMAGES);
	    }
	}
      
      //now take average image.
      for (int y = 0; y < Y_SIZE; y++)
	{
	  for (int x = 0; x < X_SIZE; x++)
	    {
	      average.at<uchar>(x,y) = image_values[X_SIZE * INIT_IMAGES * y + INIT_IMAGES * x + INIT_IMAGES / 2];
	    }
	}

      std::cout << "Init time: " << (std::clock() - start_outer) / (double) CLOCKS_PER_SEC << std::endl;
      
      // Done initializing background ---------------------------------------------------
      
      // inner loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      for(;i < SAVE_RATE; i++)
	{
	  cap >> dst;
	  
	  if (dst.empty() || dst.rows == 0 || dst.cols == 0)
	    break;
	  
	  // Perform perspective transformation
	  cv::warpPerspective(dst,frame,transform_matrix,cv::Size(X_SIZE,Y_SIZE));

	  cv::cvtColor(frame,fgMask,CV_BGR2GRAY);
	  //background subtraction
	  cv::absdiff(average,fgMask,fgMask);
	  
	  //preprocessing to conglomerate vehicle blobs -------------------------------------------------
	  Erosion(fgMask,3);
	  // dilated 3 for cam 71
	  Dilation(fgMask,4);
	  cv::threshold(fgMask,fgMask,MIN_THRESH,255,3);
	  //preprocessing complete ----------------------------------------------------------------------

	  if (i% DETECT_RATE == DETECT_RATE / 2)
	    {
	      for (unsigned int j = 0; j < objects.size(); j++)
		{
		  vehicles.at(j).second_x = objects.at(j).x;
		}
	    }
	  
	  //Going through tracked objects
	  for (unsigned int j =0; j < objects.size(); j++)
	    {
	      /// Update each tracked object using mean shift?
	      cv::meanShift(fgMask,objects.at(j),criteria);
	      objects.at(j) = objects.at(j) & cv::Rect(0,0,frame.cols, frame.rows);

	      //count passed motorcycles
	      if((objects.at(j).y >= (2*frame.rows/3) ) && vehicles.at(j).type)
		{		  
		  //filled with dummy values to prevent out of bounds error
		  //for (int k = previousVehicles.size(); k<= j; k++)
		  //previousVehicles.push_back(true);

		  if( !vehicles.at(j).over_line )
		    {
		      count++;
		      //std::cout << "Number of Motorcycles:   " << count << std::endl;
		      if (objects.at(j).x  <= leftX)
			{
			  leftcount++;
			}
		      if (objects.at(j).x  >= rightX)
			{
			  rightcount++;
			}
		      //check for proximity to lane lines.
		      for(int k = 0; k < number_lanes;k++)
			{
			  if (objects.at(j).x - lanes.at(k) < LANE_THRESH && objects.at(j).x - lanes.at(k) > -1 * LANE_THRESH && vehicles.at(j).initial_x - lanes.at(k) < LANE_THRESH && vehicles.at(j).initial_x - lanes.at(k) > -1 * LANE_THRESH)
			    {
			      interlanecount++;
			      cv::rectangle(frame, objects.at(j), cv::Scalar(0,255,0),2,1);
			    }
			}

		      //check for zigzag behavior
		      if ( vehicles.at(j).type && ( (vehicles.at(j).second_x - vehicles.at(j).initial_x) * (objects.at(j).x - vehicles.at(j).second_x) < -10))
			{
			  zigzagcount++;
			}
		    }
		  vehicles.at(j).over_line = true;
		}
	      
	      //measuring velocity of detected vehicles. 
	      if (i % DETECT_RATE == 0 && (objects.at(j).y + objects.at(j).height < frame.rows -4))
		{
		  //by clock speed.
		  double velocity = 3600 * ((double) cap.get(CV_CAP_PROP_FPS) / (double) DETECT_RATE) * (objects.at(j).y - vehicles.at(j).initial_y ) * length_kmeters / frame.rows;
		  //double velocity = 3600 * ((std::clock() - start_inner)  / (double) CLOCKS_PER_SEC) * (objects.at(j).y - initial_pos.at(j)) * length_kmeters / (double)frame.rows;
		  //discard false positives and vehicles lost by the tracker
		  if (velocity > MIN_VELOCITY && vehicles.at(j).type)
		    {
		      //std::cout << "Velocity: " << velocity << std::endl;
		      moto_velocities.push_back(velocity);
		    }
		  //Not motorycle
		  else if (velocity > MIN_VELOCITY)
		    {
		      car_velocities.push_back(velocity);
		    }
		}
	      
	      //drawing tracked rectangle on original image
	      //motorcycle
	      if (vehicles.at(j).type)
		{
		  cv::rectangle(frame, objects.at(j), cv::Scalar(255,0,0),2,1);
		  for(int k = 0; k < number_lanes;k++)
		    {
		      if (objects.at(j).x - lanes.at(k) < LANE_THRESH && objects.at(j).x - lanes.at(k) > -1 * LANE_THRESH && vehicles.at(j).initial_x - lanes.at(k) < LANE_THRESH && vehicles.at(j).initial_x - lanes.at(k) > -1 * LANE_THRESH)
			{
			  cv::rectangle(frame, objects.at(j), cv::Scalar(0,255,0),2,1);
			}
		    } 
		}
	      //other vehicles
	      else
		{
		  cv::rectangle(frame, objects.at(j), cv::Scalar(0,0,255),2,1);
		}
	    }
	  
	  if (i % DETECT_RATE == 0)
	    { 
	      vehicles.clear();
	      objects.clear();
	      
	      Detect(fgMask,objects,vehicles);
	      start_inner = std::clock();
	    }

	  //showing the mask for debugging and optimization of background subtraction parameters
	  cv::imshow("Mask",fgMask);
	  cv::warpPerspective(frame,dst,inverse_matrix,cv::Size(dst.cols,dst.rows));
	  cv::imshow("frame",dst); 
	  
	  //space for pause, escape or q to quit
	  key = cv::waitKey(1);
	  if(key == KEY_SPACE)
	    key = cv::waitKey(0);
	  if(key == KEY_ESC || key == 'q')
	    break;
	  
	}

      //using the frame rate to estimate total elapsed time.
      //duration = i / (double) cap.get(CV_CAP_PROP_FPS);

      //using the clock rate to estimate total elapsed time.
      duration = (std::clock() - start_outer) / (double) CLOCKS_PER_SEC;
      
      //calculate average velocity
      double average_moto_velocity = 0;
      double average_car_velocity = 0;
      for(i = 0; i < moto_velocities.size(); i++)
	{
	  average_moto_velocity += moto_velocities.at(i);
	}
      average_moto_velocity /= (double) moto_velocities.size();
      std::cout << "Average motorcycle velocity, km/h : " << average_moto_velocity << std::endl;
      
      for(i = 0; i< car_velocities.size(); i++)
	{
	  average_car_velocity += car_velocities.at(i);
	}
      average_car_velocity /= (double) car_velocities.size();
      
      std::cout << "Average other vehicle velocity, km/h : " << average_car_velocity << std::endl;
      
      std::cout << "Motorcycles per hour: " << 3600 * count / duration << std::endl;
      
      std::cout << "Total number of motorcycles: " << count << std::endl;
      
      std::cout << "Number of motorcycles in left prohibited zone: " << leftcount << std::endl;
      
      std::cout << "Number of motorcycles in right prohibited zone: " << rightcount << std::endl;

      std::cout << "Number of motorcycles in between lanes: " << interlanecount << std::endl;

      std::cout << "Number of zig zaging motorcycles: " << zigzagcount << std::endl;

      if (dst.empty() || dst.rows == 0 || dst.cols == 0)
	break;

      if(key == KEY_ESC || key == 'q')
	break;
      
    }
  delete[] image_values;
  cap.release();
  cvDestroyAllWindows();
  return 0;
}

void Erosion(cv::Mat &img, int radius)
{
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(2*radius+1,2*radius+1),cv::Point(radius,radius));
  cv::erode(img,img,element);
}

void Dilation(cv::Mat &img, int radius)
{
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(2*radius+1,2*radius+1),cv::Point(radius,radius));
  cv::dilate(img,img,element);
}

void Detect(cv::Mat &img, std::vector<cv::Rect> &objects, std::vector<vehicle> &vehicles)
{
  cv::Mat canny_output;
  cv::Rect ROI;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::Canny(img, canny_output, 100, 200, 3);
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3));
  cv::morphologyEx(canny_output, canny_output, cv::MORPH_CLOSE, element);
  cv::findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0) );

  for(int i =0; i < contours.size(); i++ )
    {
      if (arcLength(contours.at(i),true) > 50)
	{ 
	  ROI = cv::boundingRect(contours.at(i));
	  
	  //only choosing boxes of desired size. First for motorcycles
	  if (ROI.width < MAX_WIDTH && ROI.height > MIN_HEIGHT && ROI.width > MIN_WIDTH && ROI.height < MAX_HEIGHT && ROI.width < (ROI.height+15))
	    {
	      objects.push_back(ROI);
	      if (ROI.y > 2 * img.rows / 3)
		vehicles.push_back({ROI.y,ROI.x,ROI.x,true,true});
	      else
		vehicles.push_back({ROI.y,ROI.x,ROI.x,true,false});
	    }
	  //now checking for larger vehicles
	  else if(ROI.width >= MAX_WIDTH && ROI.height > MIN_HEIGHT)
	    {
	      objects.push_back(ROI);
	      if (ROI.y > 2 * img.rows / 3)
		vehicles.push_back({ROI.y,ROI.x,ROI.x,false,true});
	      else
		vehicles.push_back({ROI.y,ROI.x,ROI.x,false,false});
	    }
	}
    }
}

//for drawing the desired area on the image.
void onMouse(int event, int x, int y, int f,  void*)
{
  if (event == CV_EVENT_LBUTTONDOWN)
    {
      source_points[pointnumber].x = x;
      source_points[pointnumber].y = y;
      pointnumber++;
    }
}

void leftShoulder(int event, int x, int y, int f,  void*)
{
  switch(event)
    {
    case CV_EVENT_LBUTTONDOWN :
      clicked = true;
      leftX = x;
      break;
    case CV_EVENT_LBUTTONUP :
      clicked = false;
      leftX = x;
      break;
    case CV_EVENT_MOUSEMOVE :
      if (clicked)
	{
	  leftX = x;
	}
      break;
    default : break;
    }
}

void rightShoulder(int event, int x, int y, int f,  void*)
{
  switch(event)
    {
    case CV_EVENT_LBUTTONDOWN :
      clicked = true;
      rightX = x;
      break;
    case CV_EVENT_LBUTTONUP :
      clicked = false;
      rightX = x;
      break;
    case CV_EVENT_MOUSEMOVE :
      if (clicked)
	{
	  rightX = x;
	}
      break;
    default : break;
    }
}

void drawLane(int event, int x, int y, int f,  void*)
{
  switch(event)
    {
    case CV_EVENT_LBUTTONDOWN :
      clicked = true;
      lanes.at(lanenumber) = x;
      break;
    case CV_EVENT_LBUTTONUP :
      clicked = false;
      lanes.at(lanenumber) = x;
      break;
    case CV_EVENT_MOUSEMOVE :
      if (clicked)
	{
	  lanes.at(lanenumber) = x;
	}
      break;
    default : break;
    }
}


