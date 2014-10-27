#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>

using namespace std;
using namespace cv;


Mat im_src, im_gray;
vector<Point> bifur;
vector<int> bifur_eucli;

vector<Point> termin;
vector<int> termin_eucli;

void callback(int event, int x, int y, int, void*);
double eucli(Point a, Point b);

int main(int argc, char** argv)
{
	im_src = imread( argv[1],1);
	namedWindow("source",1);
	
	cvtColor(im_src, im_gray, CV_BGR2GRAY);
	threshold(im_gray, im_gray, 100, 255, cv::THRESH_BINARY);
	
	int j = im_src.rows/2;
	
	int total=0;
	
	for (int i=1; i<(im_gray.rows-1); i++)
	{
		for (int j=1; j<(im_gray.cols-1); j++)
		{

			if (im_gray.at<uchar>(Point(j,i)) == 0)
			{
				int sum = 0;
				
				int p1 = im_gray.at<uchar>(Point(j-1,i-1));
				int p2 = im_gray.at<uchar>(Point(j,i-1));
				int p3 = im_gray.at<uchar>(Point(j+1,i-1));
				int p4 = im_gray.at<uchar>(Point(j+1,i));
				int p5 = im_gray.at<uchar>(Point(j+1,i+1));
				int p6 = im_gray.at<uchar>(Point(j,i+1));
				int p7 = im_gray.at<uchar>(Point(j-1,i+1));
				int p8 = im_gray.at<uchar>(Point(j-1,i));
				
				int c1 = ((p1 == 0) && (p2 == 0))? 255 : p1;
				int c2 = ((p2 == 0) && (p3 == 0))? 255 : p2;
				int c3 = ((p3 == 0) && (p4 == 0))? 255 : p3;
				int c4 = ((p4 == 0) && (p5 == 0))? 255 : p4;
				int c5 = ((p5 == 0) && (p6 == 0))? 255 : p5;
				int c6 = ((p6 == 0) && (p7 == 0))? 255 : p6;
				int c7 = ((p7 == 0) && (p8 == 0))? 255 : p7;
				int c8 = ((p8 == 0) && (p1 == 0))? 255 : p8;
				
				//sum = (p1+p2+p3+p4+p5+p6+p7+p8)/255;
				sum = (c1+c2+c3+c4+c5+c6+c7+c8)/255;
			
				switch (sum)
				{
					case 5:
						printf("bifurcation at = x:%d y:%d [p1:%d p2:%d p3:%d p4:%d p5:%d p6:%d p7:%d p8:%d]\n",j,i,p1,p2,p3,p4,p5,p6,p7,p8);
						circle(im_src, Point(j,i),3,Scalar(0,0,255),1,8,0);
						bifur.push_back(Point(j,i));
						total++;
						break;
				
					case 7:
						printf("termination at = x:%d y:%d [p1:%d p2:%d p3:%d p4:%d p5:%d p6:%d p7:%d p8:%d]\n",j,i,p1,p2,p3,p4,p5,p6,p7,p8);
						circle(im_src, Point(j,i),3,Scalar(255,0,0),1,8,0);
						termin.push_back(Point(j,i));
						total++;
						break;
					
					default:
						break;
				}		
			}
		}
	}
	

// Cari titik pusat, ideku sih cari minimum total jarak masing2 bifur, soale lake kan ada 2 buah bifur :v
// Cari jarak masing minutiae bifurcation ke titik pusat
	
	int min = 1000000000;
	Point cpoint;
	
	for ( int i=0; i<bifur.size(); i++)
	{
		int sum=0;
		for ( int j=0; j<bifur.size(); j++)
		{
			if (i==j) { continue;}
			sum+=eucli(bifur[i],bifur[j]);
		}
		
		if (sum < min) { min = sum; cpoint=bifur[i]; }
	}
	printf("total:%d\n",total);
	printf("center point, x:%d y:%d \n",cpoint.x,cpoint.y);
	circle(im_src, cpoint, 4, Scalar(0,255,0),2,8,0);
	
// Cari jarak masing minutiae terminator ke titik pusat
	
	cout << "Euclidean : terminator minutiea" << endl;
	for ( int i=0; i<termin.size(); i++)
	{
		termin_eucli.push_back(eucli(termin[i],cpoint));
	}
	
// Cari jarak masing minutiae bifurcation ke titik pusat
	
	cout << "Euclidean : Bifurcation minutiea" << endl;
	for ( int i=0; i<bifur.size(); i++)
	{
		bifur_eucli.push_back(eucli(bifur[i],cpoint));
	}
	
//write to file
	ofstream database ("./data.txt");
	if (database.is_open())
	{
		database << termin.size() << "," << bifur.size() << "," << cpoint << "\n";
		for (int i=0; i<termin.size(); i++){ database << termin_eucli[i] << ","<< termin[i] << "\n"; }
		for (int i=0; i<bifur.size(); i++){ database << bifur_eucli[i] << ","<< termin[i] << "\n"; }
		database.close();
	}	
	
	while(1)
	{
		imshow("source",im_src);
		setMouseCallback("source",callback,0);
	
		waitKey(0);
	
		return 0;
	}	
}


double eucli(Point a, Point b)
{
    double x = a.x - b.x;
    double y = a.y - b.y;
    double dist;

    dist = pow(x,2)+pow(y,2);        
    dist = sqrt(dist);        

    return dist;
}


void callback(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		int p1 = im_gray.at<uchar>(Point(x-1,y-1));
		int p2 = im_gray.at<uchar>(Point(x,y-1));
		int p3 = im_gray.at<uchar>(Point(x+1,y-1));
		int p4 = im_gray.at<uchar>(Point(x+1,y));
		int p5 = im_gray.at<uchar>(Point(x+1,y+1));
		int p6 = im_gray.at<uchar>(Point(x,y+1));
		int p7 = im_gray.at<uchar>(Point(x-1,y+1));
		int p8 = im_gray.at<uchar>(Point(x-1,y));
		
		printf("at = x:%d y:%d [p1:%d p2:%d p3:%d p4:%d p5:%d p6:%d p7:%d p8:%d]\n",x,y,p1,p2,p3,p4,p5,p6,p7,p8);
	}
}
