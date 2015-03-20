#include <stdlib.h>
#include <sstream>
#include <stdio.h>

// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/io/pcd_io.h>
#include <vtkRenderWindow.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>
//#include <angles/angles.h>

#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>

#include <pcl/surface/convex_hull.h>
#include <pcl/surface/mls.h>

#include <pcl/filters/crop_hull.h>
#include <pcl/filters/sampling_surface_normal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>

#include <pcl/features/normal_3d.h>

#include <pcl/segmentation/crf_normal_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <boost/thread.hpp>

#include <vtkPolyLine.h>

/*
  This file aligns point clouds with each other by rejecting correspondences with RANSAC
  and then fine tuning the alignement with .
*/


typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> ColorHandlerR;
typedef pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> ColorHandlerRGB;

using namespace std;

boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer2 (new pcl::visualization::PCLVisualizer ("3D Viewer2"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));



double _distanceThreshold = 3.0;
double _pointColorThreshold = 3.0;
double _regionColorThreshold = 3.0;
double _minClusterSize = 1000;

void region_growing_rgb(pcl::PointCloud<PointT>::Ptr cloud);

void initPCLViewer(){
    //PCL Viewer
    //pclViewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&pclViewer);
    pclViewer->setBackgroundColor (0, 0, 0);
    pclViewer->initCameraParameters ();
    pclViewer->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow = pclViewer->getRenderWindow();
    renderWindow->SetSize(800,450);
    renderWindow->Render();

    pclViewer2->setBackgroundColor (0, 0, 0);
    pclViewer2->initCameraParameters ();
    pclViewer2->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow2 = pclViewer2->getRenderWindow();
    renderWindow2->SetSize(800,450);
    renderWindow2->Render();

}

pcl::PointCloud<PointT>::Ptr downsample(pcl::PointCloud<PointT>::Ptr cloud, double voxel_size){
    pcl::ScopeTime t("Voxel_grid Filtering");
    pcl::PointCloud<PointT>::Ptr returnCloud(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> grid;
    const float leaf = voxel_size;
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (cloud);
    grid.filter (*returnCloud);

    return returnCloud;
}


pcl::PointCloud<PointT>::Ptr cropAndSegmentScene(pcl::PointCloud<PointT>::Ptr scene_cloud, pcl::PointCloud<PointT>::Ptr model_cloud){

    pcl::ConvexHull<PointT> hull;
    pcl::PointCloud<PointT>::Ptr surface_hull (new pcl::PointCloud<PointT>);
    hull.setInputCloud(model_cloud);
    hull.setDimension(3);
    std::vector<pcl::Vertices> polygons;
    hull.reconstruct(*surface_hull, polygons);

    pcl::PointCloud<PointT>::Ptr objects (new pcl::PointCloud<PointT>);
    pcl::CropHull<PointT> bb_filter2;
    bb_filter2.setDim(3);
    bb_filter2.setInputCloud(scene_cloud);
    bb_filter2.setHullIndices(polygons);
    bb_filter2.setHullCloud(surface_hull);
    bb_filter2.filter(*objects);

//    pclViewer->addPointCloud (objects, ColorHandlerT(objects, 0.0, 255.0, 0.0), "scene_filtered");
//    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_filtered");

    return objects;
}

void region_growing_rgb_thread(pcl::PointCloud<PointT>::Ptr cloud){

    region_growing_rgb(cloud);

    bool continueLoop = true;

    while(continueLoop){

        std::cout << continueLoop  << std::endl;

        std::cout << "Choose one of the parameters to modify and press enter : "     << std::endl;
        std::cout << "1) Distance Threshold (default = 3.0)"                         << std::endl;
        std::cout << "2) Point Color Threshold (defaul t = 3.0)"                     << std::endl;
        std::cout << "3) Region Color Threshold (default = 3.0) "                    << std::endl;
        std::cout << "4) Minimum Cluster Size (default = 1000) "                     << std::endl;
        std::cout << "5) Quit"                                                       << std::endl;

        string selection = "";
        std::cin >> selection;

        if(selection == "5") {
            break;
        }

        std::cout << "Enter the desired value : " << std::endl;
        double parameter_value = 0.0;
        std::string parameter_value_str = "";
        std::cin >> parameter_value_str;
        parameter_value = atof(parameter_value_str.c_str());

        if(selection == "1") {
            _distanceThreshold = parameter_value;
            std::cout << "Value set!" << std::endl;
        }
        else if(selection == "2"){
            _pointColorThreshold = parameter_value;
            std::cout << "Value set!" << std::endl;
        }
        else if(selection == "3"){
            _regionColorThreshold = parameter_value;
            std::cout << "Value set!" << std::endl;
        }
        else if(selection == "4"){
            _minClusterSize = parameter_value;
            std::cout << "Value set!" << std::endl;
        }
        else{
            continueLoop = false;
        }

        region_growing_rgb(cloud);

    }

}

void region_growing_rgb(pcl::PointCloud<PointT>::Ptr cloud){
    pcl::ScopeTime t("region_growing_rgb");

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::RegionGrowingRGB<PointT> reg;
    reg.setInputCloud (cloud);
    reg.setSearchMethod (tree);
    reg.setDistanceThreshold (_distanceThreshold);
    reg.setPointColorThreshold (_pointColorThreshold);
    reg.setRegionColorThreshold (_regionColorThreshold);
    reg.setMinClusterSize (_minClusterSize);

    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();

    pclViewer->removeAllPointClouds();

    pclViewer->addPointCloud (colored_cloud, ColorHandlerRGB(colored_cloud), "segmentation");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "segmentation");


    //return colored_cloud;
}

void regionGrowing(pcl::PointCloud<PointT>::Ptr cloud){

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud);
    normal_estimator.setRadiusSearch(0.02);
    normal_estimator.compute (*normals);





    pcl::RegionGrowing<PointT, pcl::Normal> reg;
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (1000000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (cloud);
    //reg.setIndices (indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (pcl::deg2rad(5.0));
    reg.setCurvatureThreshold (1.0);

    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();

    pclViewer->addPointCloud (colored_cloud, ColorHandlerRGB(colored_cloud), "segmentation");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "segmentation");
}

void addSupervoxelConnectionsToViewer (pcl::PointXYZRGBA &supervoxel_center,
                                       pcl::PointCloud<pcl::PointXYZRGBA> &adjacent_supervoxel_centers,
                                       std::string supervoxel_name,
                                       boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer)
{
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();
    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();

    //Iterate through all adjacent points, and add a center point to adjacent point pair
    PointCloudT::iterator adjacent_itr = adjacent_supervoxel_centers.begin ();
    for ( ; adjacent_itr != adjacent_supervoxel_centers.end (); ++adjacent_itr)
    {
        points->InsertNextPoint (supervoxel_center.data);
        points->InsertNextPoint (adjacent_itr->data);
    }
    // Create a polydata to store everything in
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
    // Add the points to the dataset
    polyData->SetPoints (points);
    polyLine->GetPointIds  ()->SetNumberOfIds(points->GetNumberOfPoints ());
    for(unsigned int i = 0; i < points->GetNumberOfPoints (); i++)
        polyLine->GetPointIds ()->SetId (i,i);
    cells->InsertNextCell (polyLine);
    // Add the lines to the dataset
    polyData->SetLines (cells);

    viewer->addModelFromPolyData (polyData,supervoxel_name);
}

void superVoxels(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud){

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;
    typedef pcl::PointNormal PointNT;
    typedef pcl::PointCloud<PointNT> PointNCloudT;
    typedef pcl::PointXYZL PointLT;
    typedef pcl::PointCloud<PointLT> PointLCloudT;

    pcl::ScopeTime t("Segmentation in Supervoxels");

    double voxel_resolution = 0.01;
    double seed_resolution = 0.05;
    bool use_transform = true;
    double color_importance = 0.2;
    double spatial_importance = 0.4;
    double normal_importance = 1;

    pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution, use_transform);
    super.setInputCloud (cloud);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);

    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    super.extract (supervoxel_clusters);
    //super.refineSupervoxels(10, supervoxel_clusters);
    pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());


    // Voxel Cloud Colors
    PointCloudT::Ptr colored_cloud = super.getColoredVoxelCloud ();
    pclViewer->addPointCloud (colored_cloud, "colored voxels");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "colored voxels");

    // Voxel Graph
    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);


//    // Print Graph Connectivity
//    std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin ();
//    for ( ; label_itr != supervoxel_adjacency.end (); )
//    {
//        //First get the label
//        uint32_t supervoxel_label = label_itr->first;
//        //Now get the supervoxel corresponding to the label
//        pcl::Supervoxel<PointT>::Ptr supervoxel = supervoxel_clusters.at(supervoxel_label);

//        //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
//        PointCloudT adjacent_supervoxel_centers;
//        std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first;
//        for ( ; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
//        {
//            pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = supervoxel_clusters.at (adjacent_itr->second);
//            adjacent_supervoxel_centers.push_back (neighbor_supervoxel->centroid_);
//        }
//        //Now we make a name for this polygon
//        std::stringstream ss;
//        ss << "supervoxel_" << supervoxel_label;
//        //This function is shown below, but is beyond the scope of this tutorial - basically it just generates a "star" polygon mesh from the points given
//        addSupervoxelConnectionsToViewer (supervoxel->centroid_, adjacent_supervoxel_centers, ss.str (), pclViewer);
//        //Move iterator forward to next label
//        label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
//    }




}

pcl::PointCloud<PointT>::Ptr smoothPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){

    pcl::ScopeTime t("Moving Least Squares");

//    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZRGB>);
//    mls.setInputCloud (cloud);
//    mls.setSearchRadius (0.04);
//    mls.setPolynomialFit (true);
//    mls.setPolynomialOrder (2);
//    mls.process (*cloud_smoothed);


    pcl::PointCloud<PointT>::Ptr cloud_smoothed2(new pcl::PointCloud<PointT>);
    pcl::MovingLeastSquares<pcl::PointXYZRGB, PointT> mls2;
    mls2.setInputCloud (cloud);
    mls2.setSearchRadius (0.04);
    mls2.setPolynomialFit (true);
    mls2.setPolynomialOrder (2);
//    mls2.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZRGB, PointT>::SAMPLE_LOCAL_PLANE);
//    mls2.setUpsamplingRadius (0.01);
//    mls2.setUpsamplingStepSize (0.005);
    mls2.process (*cloud_smoothed2);

    cloud_smoothed2 = downsample(cloud_smoothed2, 0.01);


    pcl::PointCloud<PointT>::Ptr cloud_smoothed2_translated(new pcl::PointCloud<PointT>);
    Eigen::Matrix4f cloud_translation;
    cloud_translation.setIdentity();
    cloud_translation(0,3) = 1; //x translation to compare with other mls
    pcl::transformPointCloud(*cloud_smoothed2, *cloud_smoothed2_translated, cloud_translation);



    pclViewer2->addPointCloud (cloud, ColorHandlerRGB(cloud), "cloud");
    pclViewer2->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    pclViewer2->addPointCloud (cloud_smoothed2_translated, ColorHandlerT(cloud_smoothed2_translated, 255.0, 0.0, 0.0), "smoothed2");
    pclViewer2->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "smoothed2");




    return cloud_smoothed2;


}

int main (int argc, char** argv){


    pcl::PointCloud<PointT>::Ptr scene_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr model_cloud(new pcl::PointCloud<PointT>);

    pcl::io::loadPCDFile("../customAlignment_scene.pcd", *scene_cloud);
    pcl::io::loadPCDFile("../customAlignment_fine.pcd", *model_cloud);

    initPCLViewer();

    pcl::PointCloud<PointT>::Ptr scene_segmented(new pcl::PointCloud<PointT>);
    scene_segmented = cropAndSegmentScene(scene_cloud, model_cloud);

    // Smooth quality of object
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_segmented_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*scene_segmented, *scene_segmented_xyzrgb);

    pcl::PointCloud<PointT>::Ptr smoothed_cloud(new pcl::PointCloud<PointT>);
    smoothed_cloud = smoothPointCloud(scene_segmented_xyzrgb);

   // pcl::io::savePCDFileASCII(std::string("../toSegment.pcd"), *smoothed_cloud);

//    boost::thread regionThread(region_growing_rgb_thread, smoothed_cloud);


    // SuperVoxels
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smoothed_cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::copyPointCloud(*smoothed_cloud, *smoothed_cloud_xyzrgb);
    superVoxels(smoothed_cloud_xyzrgb);


    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce (100);
        pclViewer2->spinOnce (100);
    }

    return 0;
}
