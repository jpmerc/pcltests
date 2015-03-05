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

#include <tf/tf.h>

#include <pcl_ros/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZRGB PointT;
using namespace std;

boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
pcl::PointCloud<PointT>::Ptr in1(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr in2(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr in2_transformed(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr out(new pcl::PointCloud<PointT>);

int l_count = 0;
bool sData = true;
bool tData = true;
bool mData = false;
bool rData = false;



void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
    l_count = l_count + 1;
    if(l_count < 2){
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
        if (event.getKeySym () == "1" && tData){
            viewer->removePointCloud("in1");
            tData = false;
        }
        else if (event.getKeySym () == "1" && !tData){
            pcl::visualization::PointCloudColorHandlerCustom<PointT> yellow(in1, 255, 255, 0);
            viewer->addPointCloud<PointT>(in1, yellow ,"in1");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "in1");
            tData = true;
        }

        else if(event.getKeySym () == "2" && sData){
            viewer->removePointCloud("in2");
            sData = false;
        }
        else if(event.getKeySym () == "2" && !sData){
            //      viewer->removePointCloud("source");
            pcl::visualization::PointCloudColorHandlerCustom<PointT> red(in2_transformed, 255.0, 0.0, 0.0);
            pclViewer->addPointCloud (in2_transformed, red, "in2");
            pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "in2");
            sData = true;
        }
        else if(event.getKeySym () == "3" && mData){
            viewer->removePointCloud("merged");
            mData = false;
        }
        else if(event.getKeySym () == "3" && !mData){
            // viewer->removePointCloud("merged");
            pcl::visualization::PointCloudColorHandlerCustom<PointT> green(out, 0.0, 255.0, 0.0);
            pclViewer->addPointCloud (out, green, "merged");
            pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "merged");
            mData = true;
        }
        else if(event.getKeySym () == "4" && rData){
            viewer->removePointCloud("rgb");
            mData = false;
        }
        else if(event.getKeySym () == "4" && !rData){
            // viewer->removePointCloud("merged");
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(out);
            pclViewer->addPointCloud<pcl::PointXYZRGB>(out, rgb, "rgb");
            pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "rgb");
            mData = true;
        }

    }
    else{
        l_count = 0;
    }

}


Eigen::Matrix4f computeMatrixFromTransform(double x, double y, double z, double rx, double ry, double rz, double rw){

    Eigen::Vector3f vec(x, y, z);
    Eigen::Quaternionf quat(rw, rx, ry, rz);
    Eigen::Matrix3f rot = quat.toRotationMatrix();

    Eigen::Matrix4f mat;
    mat.setZero();
    mat.block(0, 0, 3, 3) = rot;
    mat.block(0, 3, 3, 1) = vec;
    mat(3,3) = 1;

    return mat;
}

pcl::PointCloud<PointT>::Ptr computeUniformSampling(pcl::PointCloud<PointT>::Ptr p_cloudIn, double radius)
{


    std::cout << "US computation begin" << std::endl;

    pcl::UniformSampling<PointT> uniformSampling;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    boost::shared_ptr<std::vector<int> > point_cloud_indice (new std::vector<int> ());
    pcl::PointCloud<int> point_cloud_out;
    uniformSampling.setInputCloud(p_cloudIn);
    uniformSampling.setSearchMethod(tree);
    uniformSampling.setRadiusSearch(radius);
    uniformSampling.compute(point_cloud_out);

    for(int i = 0; i < point_cloud_out.size(); i++)
    {
        point_cloud_indice->push_back(point_cloud_out.at(i));
    }


    pcl::PointCloud<PointT>::Ptr sampled_cloud (new pcl::PointCloud<PointT>);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(p_cloudIn);
    extract.setIndices(point_cloud_indice);
    extract.setNegative(false);
    extract.filter(*sampled_cloud);


    std::cout << "Pointcloud input size = " << p_cloudIn->size() << std::endl;
    std::cout << "Point cloud out size = " << point_cloud_out.size() << std::endl;
    std::cout << "Keypoints Size = " << sampled_cloud->size() << std::endl;

    return sampled_cloud;
}

int main (int argc, char** argv){


    pcl::io::loadPCDFile("/home/jp/devel/src/perception3d/src/tests/device1_1.pcd", *in1);
    pcl::io::loadPCDFile("/home/jp/devel/src/perception3d/src/tests/device2_1.pcd", *in2);



//    tf::Transform tf1;
//    tf1.setOrigin(tf::Vector3(-0.012687, 0.2937498, 1.0124953));
//    tf1.setRotation(tf::Quaternion(-0.36378023, -0.32528895, -0.56674915, 0.663812041));
//    Eigen::Matrix4f tf1_matrix;
//    pcl_ros::transformAsMatrix(tf1, tf1_matrix);


//    tf::Transform tf2;
//    tf2.setOrigin(tf::Vector3(-0.1223497, 0.28168088, 1.1013584));
//    tf2.setRotation(tf::Quaternion(-0.5120674, -0.0235908, -0.04915027, 0.85721325));
//    Eigen::Matrix4f tf2_matrix;
//    pcl_ros::transformAsMatrix(tf2, tf2_matrix);

    Eigen::Matrix4f tf1_matrix = computeMatrixFromTransform(-0.012687, 0.2937498, 1.0124953, -0.36378023, -0.32528895, -0.56674915, 0.663812041);
    Eigen::Matrix4f tf2_matrix = computeMatrixFromTransform(-0.1223497, 0.28168088, 1.1013584, -0.5120674, -0.0235908, -0.04915027, 0.85721325);


    Eigen::Matrix4f coord_transform = tf1_matrix * tf2_matrix.inverse();


    in1 = computeUniformSampling(in1, 0.01);
    in2 = computeUniformSampling(in2, 0.01);
//        pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;
//        icp.setInputSource(in2);
//        icp.setInputTarget(in1);
//        icp.setMaxCorrespondenceDistance(0.1);
//        icp.setMaximumIterations(30);
//        pcl::PointCloud<PointT>::Ptr Final(new pcl::PointCloud<PointT>());
//        icp.align(*Final,coord_transform);
//        Eigen::Matrix4f icp_transform = icp.getFinalTransformation();

    pcl::transformPointCloud(*in2, *in2_transformed, coord_transform);


    *out = *in1 + *in2_transformed;


    //PCL Viewer
    pclViewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&pclViewer);
    pclViewer->setBackgroundColor (0, 0, 0);
    pclViewer->initCameraParameters ();
    pclViewer->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow = pclViewer->getRenderWindow();
    renderWindow->SetSize(800,450);
    renderWindow->Render();


    pcl::visualization::PointCloudColorHandlerCustom<PointT> yellow(in1, 255.0, 255.0, 0.0);
    pclViewer->addPointCloud (in1, yellow, "in1");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "in1");


    pcl::visualization::PointCloudColorHandlerCustom<PointT> red(in2_transformed, 255.0, 0.0, 0.0);
    pclViewer->addPointCloud (in2_transformed, red, "in2");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "in2");

    //        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(out);
    //        pclViewer->addPointCloud<pcl::PointXYZRGB>(out,rgb,"out");
    //        pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "out");




    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce (100);
    }
    return 0;
}
