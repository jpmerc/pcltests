// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <vtkRenderWindow.h>
#include <pcl/common/transforms.h>

#include <stdlib.h>
#include <sstream>
#include <stdio.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>



typedef pcl::PointXYZRGB PointT;
using namespace std;






boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));




int main (int argc, char** argv){

    pcl::PointCloud<PointT>::Ptr in1(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr in2(new pcl::PointCloud<PointT>);

    pcl::io::loadPCDFile("../device1_1.pcd", *in1);
    pcl::io::loadPCDFile("../device2_1.pcd", *in2);


    pcl::PointCloud<PointT>::Ptr out(new pcl::PointCloud<PointT>);
    *out = *in1 + *in2;

    //pcl::transformPointCloud(*input_cloud,*transformed_cloud,sgurf_tf.at(0));

    //PCL Viewer
    //pclViewer->registerKeyboardCallback (keyboardEventOccurred2, (void*)&pclViewer);
    pclViewer->setBackgroundColor (0, 0, 0);
    pclViewer->initCameraParameters ();
    pclViewer->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow = pclViewer->getRenderWindow();
    renderWindow->SetSize(800,450);
    renderWindow->Render();


    pcl::visualization::PointCloudColorHandlerCustom<PointT> yellow(in1, 255.0, 255.0, 0.0);
    pclViewer->addPointCloud (in1, yellow, "in1");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "in1");


    pcl::visualization::PointCloudColorHandlerCustom<PointT> red(in2, 255.0, 0.0, 0.0);
    pclViewer->addPointCloud (in2, red, "in2");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "in2");

//    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(out);
//    pclViewer->addPointCloud<pcl::PointXYZRGB>(out,rgb,"out");
//    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "out");

    //Eigen::Quaternionf rot(w,x,y,z);
    Eigen::Vector3f offset(-0.012687, 0.2937498, 1.0124953);
    Eigen::Quaternionf quat(0.663812041, -0.36378023, -0.32528895, -0.56674915);
    Eigen::Matrix3f rot = quat.toRotationMatrix();

    pcl::transformPointCloud();


    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce (100);
    }
    return 0;
}
