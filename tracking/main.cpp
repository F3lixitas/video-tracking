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

cv::Rect init_window(cv::Mat frame, cv::Mat hist, cv::Mat mask, int channels[], const float* range[], cv::Size dstSize) {
    cv::Size s = frame.size();
    cv::Rect window;
    cv::TermCriteria term_crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);

    cv::Mat dst;
    cv::calcBackProject(&frame, 1, channels, hist, dst, range);
    cv::multiply(dst, mask, dst);

    int n;
    cv::Size orig(0, 0);
    for(n = 2; n < 100 && s.width/n >= dstSize.width && s.height/n >= dstSize.height; n*=2) {
        cv::Rect ts1(orig.width, orig.height, s.width/n, s.height/n);
        cv::Rect ts2(orig.width + s.width/n, orig.height, s.width/n, s.height/n);
        cv::Rect ts3(orig.width, orig.height + s.height/n, s.width/n, s.height/n);
        cv::Rect ts4(orig.width + s.width/n, orig.height + s.height/n, s.width/n, s.height/n);
        std::vector<cv::Scalar> res = {cv::sum(dst(ts1)), cv::sum(dst(ts2)), cv::sum(dst(ts3)), cv::sum(dst(ts4))};
        std::vector<double> sums;
        sums.reserve(4);
        sums.push_back(res[0][0] + res[0][1] + res[0][2] + res[0][3]);

        int max_i = 0;
        for(int i = 1; i < 4; i++) {
            sums.push_back(res[i][0] + res[i][1] + res[i][2] + res[i][3]);
            if(sums[i] > sums[max_i]) max_i = i;
        }

        orig.width += (max_i%2)*s.width/n;
        orig.height += (max_i/2)*s.height/n;
    }

    window.x = orig.width;
    window.y = orig.height;
    window.width = dstSize.width;
    window.height = dstSize.height;

    meanShift_custom(dst, window, term_crit);

    return window;
}

void plot_hist(cv::Mat hist) {
    int histSize = 100;
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
              cv::Scalar( 255, 0, 0), 2, 8, 0 );
    }
    cv::imshow("hist", histImage);
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
    inRange(hsv_roi, cv::Scalar(0, 255, 100), cv::Scalar(99, 255, 255), mask);

    float range_[] = {0, 99};
    const float* range[] = {range_};
    cv::Mat roi_hist, fgMask;
    int histSize[] = {100};
    int channels[] = {0};
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);

    plot_hist(roi_hist);
    cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();
    pBackSub->apply(frame, fgMask);

    track_window = init_window(frame, roi_hist, fgMask, channels, range, track_window.size());

    cv::TermCriteria term_crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 100, 1);



    double lambda = 3;
    capture >> frame;
    while(!frame.empty()) {
        cv::Mat hsv, dst;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        pBackSub->apply(frame, fgMask);

        cv::calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

        cv::Scalar res = cv::sum(dst(track_window));
        if(res[0] + res[1] + res[2] + res[3] < lambda) {
            track_window = init_window(frame, roi_hist, fgMask, channels, range, track_window.size());
        }

        cv::imshow("dst av mult", dst);
        cv::multiply(dst, fgMask, dst);
        meanShift_custom(dst, track_window, term_crit);

        cv::rectangle(frame, track_window, cv::Scalar(0, 0, 255), 2);
        cv::imshow("image", frame);
        cv::imshow("dst", dst);

        cv::Mat lum;
        cv::cvtColor(frame, lum, cv::COLOR_BGR2GRAY);
        //cv::imshow("image_lum", hsv.);
        cv::imshow("image_mask", fgMask);

        capture >> frame;

        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
} // donner à Ziad block matching
