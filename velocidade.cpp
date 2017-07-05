/*
This program is used to detect motorcycles on a designated area of the screen.
Click four times to designate the area of the screen to detect. Start with the
upper left and go clockwise. Then, drag the red line to indicate a prohibited 
zone to the left. Hit esc when finished. Then drag the blue line to indicated
a prohibited zone to the right. When finished, hit esc. The program with run
automatically.

Once running, hit q or esc to quit, or space to pause/unpause.

PARAMETERS / USAGE:

./velocidade [distance of area drawn]

*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <vector>

const int KEY_SPACE = 32;
const int KEY_ESC = 27;

//Size that image will be warped to.
const int X_SIZE = 350;
const int Y_SIZE = 400;

// parameters for motorcycle detection. MAX_WIDTH prevents larger objects being detected.
// MIN_HEIGHT specifies minimum height for motorcycle to be registered.
// In the future, can also be used for car detection. Cars must be GREATER than max_width, and greater than min_height
const int MAX_WIDTH = X_SIZE / 8;
const int MAX_HEIGHT = Y_SIZE / 3;
const int MIN_HEIGHT = Y_SIZE / 6;
const int MIN_WIDTH = X_SIZE / 13;

//Detects cars every __ frames. On other frames, tracks.
const int DETECT_RATE = 7;

// Perspective transformation
cv::Point2f source_points[] = {cv::Point2f(0,0),cv::Point2f(0,0),cv::Point2f(0,0),cv::Point2f(0,0)};
cv::Point2f dst_points[] = {cv::Point2f(0,0), cv::Point2f(X_SIZE,0),cv::Point2f(X_SIZE,Y_SIZE),cv::Point2f(0,Y_SIZE)};

//for counting motorcycles
std::vector <bool> passedVehicles;
std::vector <bool> previousVehicles;
//if true, then motorcycle. If not, then other vehicle.
std::vector <bool> vehicleType;

//for shoulder detection. Hold values for limits of prohibited/interest zones
int leftX = 0;
int rightX = 0;

// for use in transformation drawing callback.
int pointnumber = 0;

// Tracking. Only compatible with OPENCV 3.X
const std::string trackingalg = "MEDIANFLOW";

std::vector <cv::Rect> objects;

//initial y position of vehicles before tracking
std::vector<double> initial_pos;

// for drawing prohibited areas.
bool clicked = false;

//Method prototypes
void Erosion(cv::Mat &img,int radius);
void Contours(cv::Mat &img);
void Dilation(cv::Mat &img, int radius);
void onMouse(int event, int x, int y, int f, void*);
void leftShoulder(int event, int x, int y, int f, void*);
void rightShoulder(int event, int x, int y, int f, void*);

int main(int argc, char **argv)
{
  std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

  //for measuring time to calculate motorcycles per hour

  clock_t start;
  double duration;

  // for measuring velocities. lenth_kmeters is the length in kilometers of the detection area.
  double length_kmeters = 1.0d;
  // final_pos holds the y values of vehicles at end stage of tracking. Used for comparison.
  std::vector<double> final_pos;
  std::vector<double> moto_velocities;
  std::vector<double> car_velocities;

  // user inputted distance
  if (argc >=2)
    {
      length_kmeters = atof(argv[1])/1000.0d;
    }
  
  cv::namedWindow("Perspective transform",1);
    
  cv::Mat frame, fgMask, frame_gray, original, dst, temp;

  cv::VideoCapture cap("motorcycle.avi");
  
  cv::Ptr<cv::BackgroundSubtractor> subtractor;
  subtractor = new cv::BackgroundSubtractorMOG2();

  cv::TermCriteria criteria(2,2,2);
  
  if(!cap.isOpened())
    {
      std::cout << "video error " << std::endl;
      return -1;
    }

  //only for test sequences. Gets rid of initial watermarks.
  for(int i =0; i <500; i++)
    {
      cap >> frame;
    }

  // Capturing user input to create perspective transform
  cv::imshow("Perspective transform", frame);
  cv::setMouseCallback("Perspective transform",onMouse,NULL);
  
  while(pointnumber < 4 )
    {
      char c = cv::waitKey(1);
      if (c == KEY_ESC) break;
    }
  // Done capturing user input for perspective transform

  cv::Mat transform_matrix = cv::getPerspectiveTransform(source_points,dst_points);
  cv::warpPerspective(frame,dst,transform_matrix,cv::Size(X_SIZE,Y_SIZE));
  cv::imshow("Perspective transform",dst);

  // Creating prohibited areas -----------------------------------------------
  clicked = false;
  cv::setMouseCallback("Perspective transform",leftShoulder,NULL);

  for(;;)
    {
      temp = dst.clone();
      cv::line(temp,cv::Point(leftX,0),cv::Point(leftX,dst.rows-1),cv::Scalar(0,0,240),2);
      cv::imshow("Perspective transform", temp);
      char c = cv::waitKey(1);
      if (c == 27) break;
    }

  cv::line(dst,cv::Point(leftX,0),cv::Point(leftX,dst.rows-1),cv::Scalar(0,0,240),2);
  clicked = false;
  cv::setMouseCallback("Perspective transform",rightShoulder,NULL);

  for(;;)
    {
      temp = dst.clone();
      cv::line(temp,cv::Point(rightX,0),cv::Point(rightX,dst.rows-1),cv::Scalar(255,0,0),2);
      cv::imshow("Perspective transform", temp);
      char c = cv::waitKey(1);
      if (c == 27) break;
    }

  // Done creating  prohibited areas ---------------------------------------

  //if no distance supplied as command line argument, require user to input distance.
  if (argc < 2)
    {
      std::string distance_input = "";
      std::cout << "Input the distance of the vertical stretch in meters for velocity measuring: " << std::endl;
      std::cin >> distance_input;
      length_kmeters = atof(distance_input.c_str())/1000.0d;
      if (length_kmeters <= 0)
	{
	  length_kmeters = 1;
	}
    }

  //input key
  char key = '\0';

  //number of vehicles/motorcycles that have passed.
  int count = 0;
  //left and right prohibited areas
  int leftcount = 0;
  int rightcount = 0;

  /////////////////////////////// ???????? namespace
  start = clock();
  unsigned long i = 0;

  //allow background to initialize before starting detection ------------------------
  for (;i<180;i++)
  {
    cap >> original;

    if (original.empty() || original.rows == 0 || original.cols == 0)
      break;
      
    cv::warpPerspective(original,frame,transform_matrix,cv::Size(X_SIZE,Y_SIZE));

    subtractor->operator()(frame,fgMask,.002);
   
    cv::blur(fgMask,fgMask,cv::Size(3,3));
    Dilation(fgMask,3);  
    cv::threshold(fgMask,fgMask,130,255,CV_THRESH_BINARY);

  }
  // Done initializing background ---------------------------------------------------

  cv::namedWindow("Frame", 1);
  cv::namedWindow("Mask", 1);

  // Main loop ----------------------------------------------------------------------
  i = 0;
  while(true)
    {
      count = 0;
      leftcount =0;
      rightcount =0;
      i=0;
      for(;i < 54000;i++)
	{
	  cap >> original;
	  
	  if (original.empty() || original.rows == 0 || original.cols == 0)
	    break;
	  
	  // Perform perspective transformation
	  cv::warpPerspective(original,frame,transform_matrix,cv::Size(X_SIZE,Y_SIZE));

	  //backgroundsubtraction
	  subtractor->operator()(frame,fgMask,.002);
	  
	  //preprocessing to conglomerate vehicle blobs
	  Dilation(fgMask,3);
	  cv::blur(fgMask,fgMask,cv::Size(3,3));
	  cv::threshold(fgMask,fgMask,127,255,CV_THRESH_BINARY);
	  Dilation(fgMask,1);
	  
	  //counting vehicles that have passed the half way point
	  previousVehicles = passedVehicles;
	  passedVehicles.clear();
	  
	  //update the tracker
	  //trackers->Update(frame);

	  //Going through tracked objects
	  for (unsigned int j =0; j < objects.size(); j++)
	    {
	      /// Update each tracked object using mean shift?
	      cv::meanShift(fgMask,objects.at(j),criteria);
	      objects.at(j) = objects.at(j) & cv::Rect(0,0,frame.cols, frame.rows);

	      //count passed motorcycles
	      if(objects.at(j).y + objects.at(j).height / 2 >= frame.rows/2 && vehicleType.at(j))
		{
		  passedVehicles.push_back(true);
		  
		  // filled with dummy values to prevent out of bounds error
		  for (int k = previousVehicles.size(); k<= j; k++)
		    previousVehicles.push_back(true);
		  if( !previousVehicles.at(j) )
		    {
		      count++;
		      std::cout << "Number of Motorcycles:   " << count << std::endl;
		      if (objects.at(j).x  <= leftX)
			{
			  leftcount++;
			}
		      if (objects.at(j).x  >= rightX)
			{
			  rightcount++;
			}
		    }
		}
	      else if (vehicleType.at(j))
		{
		  passedVehicles.push_back(false);
		}
	      
	      //measuring velocity of detected vehicles. 
	      if (i % DETECT_RATE == 0 && (objects.at(j).y + objects.at(j).height < frame.rows -4))
		{
		  double velocity = 3600 * ((double) cap.get(CV_CAP_PROP_FPS) / (double) DETECT_RATE) * (objects.at(j).y + objects.at(j).height / 2 - initial_pos.at(j) ) * length_kmeters / frame.rows;
		  
		  //discard false positives and vehicles lost by the tracker
		  if (velocity > 5 && vehicleType.at(j))
		    {
		      std::cout << "Velocity: " << velocity << std::endl;
		      moto_velocities.push_back(velocity);
		    }
		  //Not motorycle
		  else if (velocity > 5)
		    {
		      car_velocities.push_back(velocity);
		    }
		}
	      
	      //drawing tracked rectangle on original image
	      //motorcycle
	      if (vehicleType.at(j))
		{
		  cv::rectangle(frame, objects.at(j), cv::Scalar(255,0,0),2,1);
		}
	      //other vehicles
	      else
		{
		  cv::rectangle(frame, objects.at(j), cv::Scalar(0,0,255),2,1);
		}
	    }
	  
	  if (i% DETECT_RATE ==0)
	    {
	      //detection using background subtraction
	      
	      //reinitialize tracked objects
	      
	      previousVehicles.clear();
	      initial_pos.clear();
	      vehicleType.clear();
	      objects.clear();
	      //trackers->release();
	      //delete trackers;
	      //trackers = new CvBlobTracker();
	      
	      Contours(fgMask);
	      
	      //trackers->addBlob(fgMask,objects);
	    }

	  //showing the mask for debugging and optimization of background subtraction parameters
	  cv::imshow("Mask",fgMask);
	  
	  cv::imshow("frame",frame); 
	  
	  //space for pause, escape or q to quit
	  key = cv::waitKey(1);
	  if(key == KEY_SPACE)
	    key = cv::waitKey(0);
	  if(key == KEY_ESC || key == 'q')
	    break;
	  
	}

      //using the frame rate to estimate total elapsed time.
      duration = i / (double) cap.get(CV_CAP_PROP_FPS);
      
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
      
      std::cout << "Average othere vehicle velocity, km/h : " << average_car_velocity << std::endl;
      
      std::cout << "Motorcycles per hour: " << 3600 * count / duration << std::endl;
      
      std::cout << "Total number of motorcycles: " << count << std::endl;
      
      std::cout << "Number of motorcycles in left prohibited zone: " << leftcount << std::endl;
      
      std::cout << "Number of motorcycles in right prohibited zone: " << rightcount << std::endl;

      if (original.empty() || original.rows == 0 || original.cols == 0)
	break;

      if(key == KEY_ESC || key == 'q')
	break;
      
    }
  subtractor.release();
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

void Contours(cv::Mat &img)
{
  cv::Mat canny_output;
  cv::Rect ROI;
  //GET RID OF VECTOR OF VECTORS FOR PERFORMANCE...
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::Canny(img, canny_output, 100, 200, 3);
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3));
  cv::morphologyEx(canny_output, canny_output, cv::MORPH_CLOSE, element);
  cv::findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0) );

  for(int i =0; i < contours.size(); i++ )
    {
      if (arcLength(contours.at(i),true) > 90)
	{ 
	  ROI = cv::boundingRect(contours.at(i));
	  
	  //only choosing boxes of desired size. First for motorcycles
	  if (ROI.width < MAX_WIDTH && ROI.height > MIN_HEIGHT && ROI.width > MIN_WIDTH && ROI.height < MAX_HEIGHT && ROI.width < (ROI.height+15))
	    {
	      objects.push_back(ROI);
	      initial_pos.push_back(ROI.y + ROI.height / 2);
	      vehicleType.push_back(true);
	    }
	  //now checking for larger vehicles
	  else if(ROI.width >= MAX_WIDTH && ROI.height > MIN_HEIGHT)
	    {
	      objects.push_back(ROI);
	      initial_pos.push_back(ROI.y + ROI.height / 2);
	      vehicleType.push_back(false);
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


