#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if(argc != 3) {
        cout << "usage: feature_extraction ../img1 ../img2" << endl;
        return 1;
    }

    //-- Read image
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    //-- Initialization
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //-- First: Detect Oriented Fast corner position
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- Second: Compute BRIEF descriptor
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB feature points", outimg1);

    //-- Third: Match descriptors between two images with Hamming distance
    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    //-- Forth: Key points filtering
    double min_dist = 10000, max_dist = 0;

    // Find the min_dist and max_dist for every two descriptors
    for(int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    cout << "-- Max dist: " << max_dist << endl;
    cout << "-- Min dist: " << min_dist << endl;

    // When the distance between the descriptors is greater than 2 * min_dist,
    // that means matching error. However, sometime the min_dist might be super
    // small. We set 30 for a threshold.
    std::vector<DMatch> good_matches;
    for(int i = 0; i < descriptors_1.rows; i++) {
        if(matches[i].distance <= max(2*min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    //-- Fifth: Draw the images
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    imshow("All key points", img_match);
    imshow("Key points after optimized", img_goodmatch);
    waitKey(0);

    return 0;
}
