#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

const char* FILENAME = "/home/felix/Videos/video_projet_aj.mp4";


int meanShift_custom( cv::InputArray _probImage, cv::Rect& window, cv::TermCriteria criteria )
{
    cv::Size size;
    int cn;
    cv::Mat mat;
    cv::UMat umat;
    bool isUMat = _probImage.isUMat();

    if (isUMat)
        umat = _probImage.getUMat(), cn = umat.channels(), size = umat.size();
    else
        mat = _probImage.getMat(), cn = mat.channels(), size = mat.size();

    cv::Rect cur_rect = window;

    CV_Assert( cn == 1 );

    if( window.height <= 0 || window.width <= 0 )
        CV_Error( cv::Error::StsBadArg, "Input window has non-positive sizes" );

    window = window & cv::Rect(0, 0, size.width, size.height);

    double eps = (criteria.type & cv::TermCriteria::EPS) ? std::max(criteria.epsilon, 0.) : 1.;
    eps = cvRound(eps*eps);
    int i, niters = (criteria.type & cv::TermCriteria::MAX_ITER) ? std::max(criteria.maxCount, 1) : 100;

    for( i = 0; i < niters; i++ )
    {
        cur_rect = cur_rect & cv::Rect(0, 0, size.width, size.height);
        if( cur_rect == cv::Rect() )
        {
            cur_rect.x = size.width/2;
            cur_rect.y = size.height/2;
        }
        cur_rect.width = std::max(cur_rect.width, 1);
        cur_rect.height = std::max(cur_rect.height, 1);

        cv::Moments m = isUMat ? moments(umat(cur_rect)) : moments(mat(cur_rect));

        // Calculating center of mass
        if( fabs(m.m00) < DBL_EPSILON )
            break;

        int dx = cvRound( m.m10/m.m00 - window.width*0.5 );
        int dy = cvRound( m.m01/m.m00 - window.height*0.5 );

        int nx = std::min(std::max(cur_rect.x + dx, 0), size.width - cur_rect.width);
        int ny = std::min(std::max(cur_rect.y + dy, 0), size.height - cur_rect.height);

        dx = nx - cur_rect.x;
        dy = ny - cur_rect.y;
        cur_rect.x = nx;
        cur_rect.y = ny;

        // Check for coverage centers mass & window
        if( dx*dx + dy*dy < eps )
            break;
    }

    window = cur_rect;
    return i;
}

int main(int argc, char* argv[]) {
    cv::VideoCapture capture(FILENAME);
    if (!capture.isOpened()){
        //error in opening the video input
        std::cerr << "Unable to open file!\n";
        return 0;
    }

    cv::Mat frame;
    capture >> frame;

    cv::Rect track_window(167, 326, 28, 28);

    cv::Mat roi = frame(track_window);
    cv::Mat hsv_roi, mask;
    cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
    inRange(hsv_roi, cv::Scalar(0, 60, 32), cv::Scalar(180, 255, 255), mask);

    float range_[] = {0, 180};
    const float* range[] = {range_};
    cv::Mat roi_hist, fgMask;
    int histSize[] = {180};
    int channels[] = {0};
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);

    cv::TermCriteria term_crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);

    cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();

    capture >> frame;
    while(!frame.empty()) {
        cv::Mat hsv, dst;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        pBackSub->apply(frame, fgMask);

        cv::calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

        cv::multiply(dst, fgMask, dst);
        meanShift_custom(dst, track_window, term_crit);

        cv::rectangle(frame, track_window, 255, 2);
        cv::imshow("image", frame);
        //cv::imshow("image_mask", fgMask);

        capture >> frame;

        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
}
