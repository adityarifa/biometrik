#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#include <iostream>
#include <stdio.h>
#include <fstream>

#define PI 3.14159265

using namespace std;
using namespace cv;

namespace GuoHall {
    void thinningIteration(cv::Mat& im, int iter)
    {
        cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

        for (int i = 1; i < im.rows; i++)
        {
            for (int j = 1; j < im.cols; j++)
            {
                uchar p2 = im.at<uchar>(i-1, j);
                uchar p3 = im.at<uchar>(i-1, j+1);
                uchar p4 = im.at<uchar>(i, j+1);
                uchar p5 = im.at<uchar>(i+1, j+1);
                uchar p6 = im.at<uchar>(i+1, j);
                uchar p7 = im.at<uchar>(i+1, j-1);
                uchar p8 = im.at<uchar>(i, j-1);
                uchar p9 = im.at<uchar>(i-1, j-1);

                int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
                int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
                int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
                int N  = N1 < N2 ? N1 : N2;
                int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

                if (C == 1 && (N >= 2 && N <= 3) & m == 0)
                    marker.at<uchar>(i,j) = 1;
            }
        }

        im &= ~marker;
    }
    void thinning(cv::Mat& im)
    {
        im /= 255;

        cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
        cv::Mat diff;

        do {
            thinningIteration(im, 0);
            thinningIteration(im, 1);
            cv::absdiff(im, prev, diff);
            im.copyTo(prev);
        }
        while (cv::countNonZero(diff) > 0);

        im *= 255;
    }
}

namespace ZhangSuen {
    void thinningIteration(cv::Mat& im, int iter)
    {
        cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

        for (int i = 1; i < im.rows-1; i++)
        {
            for (int j = 1; j < im.cols-1; j++)
            {
                uchar p2 = im.at<uchar>(i-1, j);
                uchar p3 = im.at<uchar>(i-1, j+1);
                uchar p4 = im.at<uchar>(i, j+1);
                uchar p5 = im.at<uchar>(i+1, j+1);
                uchar p6 = im.at<uchar>(i+1, j);
                uchar p7 = im.at<uchar>(i+1, j-1);
                uchar p8 = im.at<uchar>(i, j-1);
                uchar p9 = im.at<uchar>(i-1, j-1);

                int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                    marker.at<uchar>(i,j) = 1;
            }
        }
        im &= ~marker;
    }
    void thinning(cv::Mat& im)
    {
        im /= 255;

        cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
        cv::Mat diff;

        do {
            thinningIteration(im, 0);
            thinningIteration(im, 1);
            cv::absdiff(im, prev, diff);
            im.copyTo(prev);
        }
        while (cv::countNonZero(diff) > 0);

        im *= 255;
    }
}

typedef void (*algorithm_function_ptr_t)(cv::Mat&);
void originalImage(cv::Mat& source) {
    return;
}

Mat im_src, im_gray, im_segmented, im_output, im_thinned, im_roi;
vector<Point> bifur;
vector<int> bifur_eucli;

vector<Point> termin;
vector<int> termin_eucli;

void callback(int event, int x, int y, int, void*);
void Morphology_Operations();
double eucli(Point a, Point b);
float orientation(int j, int i, int idx);
void saveData();

int main(int argc, char** argv)
{
    im_src = imread( argv[1],1);
    namedWindow("source",1);

    cvtColor(im_src, im_gray, CV_BGR2GRAY);

    //--------------Image Segmentation
    int left_limit = im_gray.cols, right_limit = 0, top_limit = im_gray.rows, bottom_limit = 0;

    int segmentation_block_size = 32;
    for (int i = 0; i < im_gray.cols - segmentation_block_size; i = i + segmentation_block_size)
    {
        for (int j = 0; j < im_gray.rows - segmentation_block_size; j = j + segmentation_block_size)
        {
            double mean_val = 0;
            for (int _i = 0; _i < segmentation_block_size; _i++)
            {
                for (int _j = 0; _j < segmentation_block_size; _j++)
                {
                    mean_val = mean_val + im_gray.at<uchar>(Point(i + _i, j + _j));
                }
            }
            mean_val = mean_val/pow(segmentation_block_size, 2);

            double sub_var = 0;
            for (int _i = 0; _i < segmentation_block_size; _i++)
            {
                for (int _j = 0; _j < segmentation_block_size; _j++)
                {
                    sub_var = sub_var + pow((im_gray.at<uchar>(Point(i + _i, j + _j)) - mean_val), 2.0);
                }
            }
            sub_var = sub_var / pow(segmentation_block_size, 2.0);

            for (int _i = 0; _i < segmentation_block_size; _i++)
            {
                for (int _j = 0; _j < segmentation_block_size; _j++)
                {
                    if (sub_var > 1000)
                    {
                        if (i+_i < left_limit) left_limit = i+_i;
                        if (i+_i > right_limit) right_limit = i+_i;
                        if (j+_j < top_limit) top_limit = j+_j;
                        if (j+_j > bottom_limit) bottom_limit = j+_j;
                    }
                }
            }
        }
    }
    Rect bound = Rect(left_limit, top_limit, ceil((float)(right_limit-left_limit)/segmentation_block_size)*segmentation_block_size,ceil((float)(bottom_limit-top_limit)/segmentation_block_size)*segmentation_block_size);
    im_segmented = im_gray(bound);

    threshold(im_segmented, im_segmented, 100, 255, cv::THRESH_BINARY);

    im_thinned = im_segmented.clone();

    ZhangSuen::thinning(im_thinned);
    //im_output = im_thinned.clone();
    im_output = Mat (im_thinned.rows, im_thinned.cols, CV_8UC3, Scalar(255,255,255));

    for(int i = 0; i < im_thinned.rows; i++)
    {
        for (int j = 0; j < im_thinned.cols; j++)
        {
            if (im_thinned.at<uchar>(i, j) == 255)
            {
                im_output.at<Vec3b>(i, j)[0] = 0;
                im_output.at<Vec3b>(i, j)[1] = 0;
                im_output.at<Vec3b>(i, j)[2] = 0;
            }
            im_thinned.at<uchar>(i, j) = 255 - im_thinned.at<uchar>(i, j);
        }
    }

	
	Morphology_Operations();

    int total=0;

    for (int i=1; i<(im_thinned.rows-1); i++)
    {
        for (int j=1; j<(im_thinned.cols-1); j++)
        {
			if (im_roi.at<uchar>(Point(j,i)) == 0)
			{
            if (im_thinned.at<uchar>(Point(j,i)) == 0)
            {
                int sum = 0;
                int c[8]={};

                int p1 = im_thinned.at<uchar>(Point(j-1,i-1));
                int p2 = im_thinned.at<uchar>(Point(j,i-1));
                int p3 = im_thinned.at<uchar>(Point(j+1,i-1));
                int p4 = im_thinned.at<uchar>(Point(j+1,i));
                int p5 = im_thinned.at<uchar>(Point(j+1,i+1));
                int p6 = im_thinned.at<uchar>(Point(j,i+1));
                int p7 = im_thinned.at<uchar>(Point(j-1,i+1));
                int p8 = im_thinned.at<uchar>(Point(j-1,i));

                c[0] = ((p1 == 0) && (p2 == 0))? 255 : p1;
                c[1] = ((p2 == 0) && (p3 == 0))? 255 : p2;
                c[2] = ((p3 == 0) && (p4 == 0))? 255 : p3;
                c[3] = ((p4 == 0) && (p5 == 0))? 255 : p4;
                c[4] = ((p5 == 0) && (p6 == 0))? 255 : p5;
                c[5] = ((p6 == 0) && (p7 == 0))? 255 : p6;
                c[6] = ((p7 == 0) && (p8 == 0))? 255 : p7;
                c[7] = ((p8 == 0) && (p1 == 0))? 255 : p8;

                //sum = (p1+p2+p3+p4+p5+p6+p7+p8)/255;
                sum = (c[0]+c[1]+c[2]+c[3]+c[4]+c[5]+c[6]+c[7])/255;

                if (sum == 5)
                {
					printf("bifurcation at = x:%d y:%d [p1:%d p2:%d p3:%d p4:%d p5:%d p6:%d p7:%d p8:%d]\n",j,i,p1,p2,p3,p4,p5,p6,p7,p8);
                    circle(im_output, Point(j,i),3,Scalar(0,0,255),1,8,0);
                    
                    int sum_idx;
                    int idx;
                    for (int _i=0; _i<8; _i+=2)
                    {
						if (c[_i] == 0) { sum_idx++; idx=_i; }
					}
					
					if (sum_idx == 1)
					{
						  cout << "teta: " << orientation(j,i,idx) << endl; 
					}
					else
					{
						for (int _i=1; _i<8; _i+=2){ idx=_i; }
							cout << "teta: " << orientation(j,i,idx) << endl; 
					}   
                    
                                      
                    bifur.push_back(Point(j,i));
                    total++;
				}

				else if(sum == 7)
				{
                    printf("termination at = x:%d y:%d [p1:%d p2:%d p3:%d p4:%d p5:%d p6:%d p7:%d p8:%d]\n",j,i,p1,p2,p3,p4,p5,p6,p7,p8);
                    circle(im_output, Point(j,i),3,Scalar(255,0,0),1,8,0);
                    
                    int idx;
                    for (int _i=0; _i<8; _i++)
                    {
						if (c[_i] == 0) { idx=_i; }
					}                      
                    
                    cout << "teta: " << orientation(j,i,idx) << endl;                   
                    termin.push_back(Point(j,i));
                    total++;
				}
            }
			}
        }
    }


	// ----------------- Cari titik pusat, ideku sih cari minimum total jarak masing2 bifur, soale lake kan ada 2 buah bifur :v + Cari jarak masing minutiae bifurcation ke titik pusat
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
    circle(im_output, cpoint, 4, Scalar(0,255,0),2,8,0);

	//------------------ Cari jarak masing minutiae terminator ke titik pusat
    cout << "Euclidean : terminator minutiea" << endl;
    for ( int i=0; i<termin.size(); i++)
    {
        termin_eucli.push_back(eucli(termin[i],cpoint));
    }

	//------------------ Cari jarak masing minutiae bifurcation ke titik pusat
    cout << "Euclidean : Bifurcation minutiea" << endl;
    for ( int i=0; i<bifur.size(); i++)
    {
        bifur_eucli.push_back(eucli(bifur[i],cpoint));
    }

	//------------------ save to file
    saveData();

    while(1)
    {
        imshow("source",im_src);
        imshow("segmented", im_segmented);
        imshow("out", im_output);
        imshow("thinned", im_thinned);
        setMouseCallback("out",callback,0);

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
        int p1 = im_thinned.at<uchar>(Point(x-1,y-1));
        int p2 = im_thinned.at<uchar>(Point(x,y-1));
        int p3 = im_thinned.at<uchar>(Point(x+1,y-1));
        int p4 = im_thinned.at<uchar>(Point(x+1,y));
        int p5 = im_thinned.at<uchar>(Point(x+1,y+1));
        int p6 = im_thinned.at<uchar>(Point(x,y+1));
        int p7 = im_thinned.at<uchar>(Point(x-1,y+1));
        int p8 = im_thinned.at<uchar>(Point(x-1,y));

        printf("at = x:%d y:%d [p1:%d p2:%d p3:%d p4:%d p5:%d p6:%d p7:%d p8:%d]\n",x,y,p1,p2,p3,p4,p5,p6,p7,p8);
    }
}

void Morphology_Operations()
{
	int opening =  2;
	int closing =  3;
	int morph_size = 5;

	Mat open_element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	morphologyEx( im_segmented, im_roi, opening, open_element );
	
	Mat dil_element = getStructuringElement( MORPH_RECT, Size( 4*morph_size + 1, 4*morph_size+1 ), Point( morph_size*2, morph_size*2 ) );
	dilate( im_roi, im_roi, dil_element );
	
	//Mat close_element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	//morphologyEx( im_roi, im_roi, closing, close_element );
  
	imshow( "ROI", im_roi );
}

void saveData()
{
	ofstream database ("./data.txt");
    if (database.is_open())
    {
        database << termin.size() << "," << bifur.size() << "," << "\n";
        for (int i=0; i<termin.size(); i++){ database << termin_eucli[i] << ","<< termin[i] << "\n"; }
        for (int i=0; i<bifur.size(); i++){ database << bifur_eucli[i] << ","<< termin[i] << "\n"; }
        database.close();
    }
}

float orientation(int j, int i, int idx)
{	
	float teta;
	Point nowPoint, oldPoint, newPoint;
	
	int sumx = 0;
	int sumy = 0;
	
	switch (idx)
    {
		case 0:
			oldPoint = Point(j,i);
			newPoint = Point(j-1,i-1);
			break;
							
		case 1:
			oldPoint = Point(j,i);
			newPoint = Point(j,i-1);
			break;
							
		case 2:
			oldPoint = Point(j,i);
			newPoint = Point(j+1,i-1);
			break;
							
		case 3:
			oldPoint = Point(j,i);
			newPoint = Point(j+1,i);
			break;
							
		case 4:
			oldPoint = Point(j,i);
			newPoint = Point(j+1,i+1);
			break;
								
		case 5:
			oldPoint = Point(j,i);
			newPoint = Point(j,i+1);
			break;
							
		case 6:
			oldPoint = Point(j,i);
			newPoint = Point(j-1,i+1);
			break;
								
		case 7:
			oldPoint = Point(j,i);
			newPoint = Point(j-1,i);
			break;
								
		default:
			break;
	}
	
	for (int idx=0; idx<1; idx++) //how much tracking walk
	{
		for (int i=(newPoint.y-1); i<(newPoint.y+2); i++) // check each neighbord pixel
		{
			for (int j=(newPoint.x-1); j<(newPoint.x+2); j++)
			{
				nowPoint = Point(j,i);
				//cout << nowPoint << endl;
				if ((nowPoint != oldPoint) && (nowPoint != newPoint))
				{
					if (im_thinned.at<uchar>(nowPoint) == 0)
					{
						oldPoint = newPoint;
						newPoint = nowPoint;
						sumx+=j;
						sumy+=i;
						//cout << oldPoint << " " << nowPoint << " " << newPoint << endl;
						break;
					}
				} 	
			}
		}	
	}
	teta = atan((float)sumy/sumx) * 180 / PI;
	return teta;
}

