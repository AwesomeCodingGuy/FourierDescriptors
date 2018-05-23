// Qt
#include <QCoreApplication>
#include <QDebug>
#include <QChartView>
#include <QLineSeries>
#include <QMainWindow>

// STL
#include <iostream>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

QT_CHARTS_USE_NAMESPACE

// Color definitions
cv::Scalar colors[3] =
{
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255)
};

// Contour data
struct ContourData
{
    const int approxLength = 32;
    int idx;
    std::string filename;
    std::string windowName;
    cv::Mat source;
    cv::Mat sourceBinary;
    cv::Mat contourImage;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> contour;
    std::vector<cv::Point2f> approxContour;
    cv::Point2f centroidPos;

    QLineSeries *lineSeries;
};

int approximateContour(std::vector<cv::Point> &input, std::vector<cv::Point2f> &output, int length)
{
    float idx;
    float f, f0, f1;
    if(length < 0)
        return -1;
    output.resize(length);

    for(int i = 0; i < length; ++i) {
        f = i / float(length);
        idx = (int)input.size() * f;
        f1 = std::modff(idx, &f0);
        output[i].x = input[f0].x * f1 + input[f0 + 1].x * (1 - f1);
        output[i].y = input[f0].y * f1 + input[f0 + 1].y * (1 - f1);
    }

    return 0;
}

int setLineSeries(QLineSeries *lineSeries, const std::vector<cv::Point2f> &input)
{
    for(const cv::Point2f &p : input)
        lineSeries->append(p.x, p.y);

    return 0;
}

int drawContourPoints(cv::Mat &image, const std::vector<cv::Point2f> &contour, const cv::Scalar &color)
{
    for(int i = 0; i < contour.size(); ++i) {
        cv::circle(image, contour[i], 1, color);
    }
    return 0;
}

cv::Point2f calculateCentroid(const std::vector<cv::Point2f> &contour) {
    cv::Moments moments = cv::moments(contour);
    float cX = moments.m10 / moments.m00;
    float cY = moments.m01 / moments.m00;

    return cv::Point2f(cX, cY);
}

int main(int argc, char *argv[])
{
    // Qt Application object
    QCoreApplication a(argc, argv);
    const std::string baseFolder = "../data/sans-serif/";

    // ContourObject
    std::vector<ContourData> data;
    data.resize(10);

    // Load images
    qDebug() << "Loading images...";
    for(int i = 0; i < 10; ++i) {
        data[i].idx         = i;
        data[i].filename    = baseFolder + std::to_string(i) + ".png";
        data[i].windowName  = data[i].filename;
        data[i].source      = cv::imread(data[i].filename);
        data[i].lineSeries  = new QtCharts::QLineSeries(&a);

        cv::namedWindow(data[i].windowName, cv::WINDOW_NORMAL);
        cv::imshow(data[i].windowName, data[i].source);
    }
    cv::waitKey(0);

    // Calculate contours
    QChart *chart = new QChart();
    chart->legend()->hide();
    for(ContourData &d : data) {
        // init
        d.sourceBinary = d.source.clone();
        d.contourImage = cv::Mat(d.source.size(), CV_8UC3, cv::Scalar(0, 0, 0));

        // convert and find contours
        cv::cvtColor(d.sourceBinary, d.sourceBinary, CV_BGR2GRAY);
        cv::threshold(d.sourceBinary, d.sourceBinary, 128, 255, CV_THRESH_BINARY);
        cv::findContours(d.sourceBinary, d.contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

        // draw main contour
        for(size_t idx = 0; idx < d.contours.size(); ++idx) {
            // only use contour at index 1 - 0 is the contour of the image border
            if(idx == 1) {
                cv::drawContours(d.contourImage, d.contours, (int)idx, colors[idx % 3]);
                d.contour = d.contours[1];
                qDebug() << "Contour Length (" << d.idx << "): " << d.contour.size();
            }
        }

        // approximate contour
        approximateContour(d.contour, d.approxContour, d.approxLength);

        // add line series
        setLineSeries(d.lineSeries, d.approxContour);

        // draw approximated contour points
        drawContourPoints(d.contourImage, d.approxContour, colors[2]);

        // calculate centroid
        d.centroidPos = calculateCentroid(d.approxContour);
        cv::circle(d.contourImage, d.centroidPos, 1, colors[0]);

        // show
        cv::imshow(d.windowName, d.contourImage);

        // show line chart
        chart->addSeries(d.lineSeries);
        chart->createDefaultAxes();
    }
    cv::waitKey(0);

    cv::destroyAllWindows();
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    QMainWindow window;
    window.setCentralWidget(chartView);
    window.resize(400, 300);
    window.show();
    // return 0;
    return a.exec();
}
