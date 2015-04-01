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


#include <pcl/filters/sampling_surface_normal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_hull.h>

#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/crf_normal_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/features/normal_3d.h>

#include <boost/thread.hpp>
#include <vtkPolyLine.h>

/*
  This file segments the aligned point clouds into objects
*/


typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> ColorHandlerR;
typedef pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> ColorHandlerRGB;

using namespace std;

boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer2 (new pcl::visualization::PCLVisualizer ("3D Viewer2"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer3 (new pcl::visualization::PCLVisualizer ("3D Viewer3"));



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

    pclViewer3->setBackgroundColor (0, 0, 0);
    pclViewer3->initCameraParameters ();
    pclViewer3->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow3 = pclViewer3->getRenderWindow();
    renderWindow3->SetSize(800,450);
    renderWindow3->Render();

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


pcl::PointCloud<PointT>::Ptr extractPlane(pcl::PointCloud<PointT>::Ptr cloud){

    pcl::PointCloud<PointT>::Ptr returned_cloud(new pcl::PointCloud<PointT>);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inlierIndices(new pcl::PointIndices);

    // Find the plane coefficients from the model
    pcl::SACSegmentation<PointT> segmentation;
    segmentation.setInputCloud(cloud);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(0.02);
    segmentation.setOptimizeCoefficients(true);
    segmentation.segment(*inlierIndices, *coefficients);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inlierIndices);
    extract.filter (*returned_cloud);

    return returned_cloud;

}

pcl::PointCloud<PointT>::Ptr getCentroid(pcl::PointCloud<PointT>::Ptr cloud){
    pcl::PointCloud<PointT>::Ptr centroid_cloud (new pcl::PointCloud<PointT>);
    Eigen::Vector4f c;
    pcl::compute3DCentroid<PointT>(*cloud,c);
    PointT pt;
    pt.x = c[0];
    pt.y = c[1];
    pt.z = c[2];
    centroid_cloud->push_back(pt);
    return centroid_cloud;
}

pcl::PointCloud<PointT>::Ptr getCentroid2D(pcl::PointCloud<PointT>::Ptr cloud){

    pcl::PointCloud<PointT>::Ptr bottom_plane = extractPlane(cloud);

    pcl::PointCloud<PointT>::Ptr centroid_cloud (new pcl::PointCloud<PointT>);
    Eigen::Vector4f c;
    pcl::compute3DCentroid<PointT>(*bottom_plane,c);
    PointT pt;
    pt.x = c[0];
    pt.y = c[1];
    pt.z = c[2];
    centroid_cloud->push_back(pt);
    return centroid_cloud;
}
pcl::PointCloud<PointT>::Ptr shrinkCloud(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr centroid, double shrink_distance){

    // Theory
    // The equation of a 3d line is [xt;yt;zt] = [x1;y1;z1] + t [x2-x1;y2-y1;z2-z1]
    // In this case, 't' is a ratio between the distance to shrink and the distance between P2 (centroid) and P1 (pointcloud points)

    pcl::PointCloud<PointT>::Ptr shrinked_cloud (new pcl::PointCloud<PointT>);

    PointT centroid_pt = centroid->at(0);
    Eigen::Vector3f centroid_vec(centroid_pt.x, centroid_pt.y, centroid_pt.z);

    for(int i=0; i < cloud->size(); i++){

        PointT pointcloud_pt = cloud->at(i);
        Eigen::Vector3f pointcloud_vec(pointcloud_pt.x, pointcloud_pt.y, pointcloud_pt.z);

        // Calculate the difference vector and the distance between the points
        Eigen::Vector3f diff = centroid_vec - pointcloud_vec;
        double distance_between_points = diff.squaredNorm();

        // Calculate the distance ratio (shrink_distance over the distance between the points)
        double ratio = shrink_distance / distance_between_points;

        // Calculate the new coordinates of the point
        Eigen::Vector3f new_coords = pointcloud_vec + ratio * diff;

        // Assign the new coordinates to the original point
        pointcloud_pt.x = new_coords[0];
        pointcloud_pt.y = new_coords[1];
        pointcloud_pt.z = new_coords[2];
        shrinked_cloud->push_back(pointcloud_pt);

    }

    return shrinked_cloud;
}

pcl::PointCloud<PointT>::Ptr cropAndSegmentScene(pcl::PointCloud<PointT>::Ptr scene_cloud, pcl::PointCloud<PointT>::Ptr model_cloud){

    pcl::ScopeTime t("Hull");

    pcl::ConvexHull<PointT> hull;
    pcl::PointCloud<PointT>::Ptr surface_hull (new pcl::PointCloud<PointT>);
    hull.setInputCloud(model_cloud);
    hull.setDimension(3);
    std::vector<pcl::Vertices> polygons;
    hull.reconstruct(*surface_hull, polygons);

    std::cout << "Vertices : " << polygons.size() <<  std::endl;
    std::cout << "Hull Cloud : " << surface_hull->size() <<  std::endl;

    pcl::PointCloud<PointT>::Ptr centroid_cloud = getCentroid(model_cloud);
    pcl::PointCloud<PointT>::Ptr shrinked_hull = shrinkCloud(surface_hull, centroid_cloud, 0.015);
    //pcl::PointCloud<PointT>::Ptr shrinked_hull = surface_hull;

    pcl::PointCloud<PointT>::Ptr objects (new pcl::PointCloud<PointT>);
    pcl::CropHull<PointT> bb_filter2;
    bb_filter2.setDim(3);
    bb_filter2.setInputCloud(scene_cloud);
    bb_filter2.setHullIndices(polygons);
    bb_filter2.setHullCloud(shrinked_hull);
    bb_filter2.filter(*objects);

    pclViewer->addPointCloud (surface_hull, ColorHandlerT(surface_hull, 0.0, 0.0, 255.0), "hull");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "hull");

    pclViewer->addPointCloud (shrinked_hull, ColorHandlerT(shrinked_hull, 0.0, 255.0, 0.0), "shrinked_hull");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "shrinked_hull");

    pclViewer->addPointCloud (scene_cloud, ColorHandlerT(scene_cloud, 255.0, 0.0, 0.0), "scene");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");

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

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud <pcl::PointXYZRGB>);
    colored_cloud = reg.getColoredCloud ();

    pclViewer->removeAllPointClouds();

    if(colored_cloud){
        std::cout << "Size = " << colored_cloud->size() << std::endl;
        //    pclViewer->addPointCloud (colored_cloud, ColorHandlerRGB(colored_cloud), "segmentation");
        //    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "segmentation");

    }
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
    pclViewer3->addPointCloud (colored_cloud, "colored voxels");
    pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "colored voxels");

    // Voxel Graph
    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);


    // Print Graph Connectivity
    std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin ();
    for ( ; label_itr != supervoxel_adjacency.end (); )
    {
        //First get the label
        uint32_t supervoxel_label = label_itr->first;
        //Now get the supervoxel corresponding to the label
        pcl::Supervoxel<PointT>::Ptr supervoxel = supervoxel_clusters.at(supervoxel_label);

        //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
        PointCloudT adjacent_supervoxel_centers;
        std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first;
        for ( ; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
        {
            pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = supervoxel_clusters.at (adjacent_itr->second);
            adjacent_supervoxel_centers.push_back (neighbor_supervoxel->centroid_);
        }
        //Now we make a name for this polygon
        std::stringstream ss;
        ss << "supervoxel_" << supervoxel_label;
        //This function is shown below, but is beyond the scope of this tutorial - basically it just generates a "star" polygon mesh from the points given
        addSupervoxelConnectionsToViewer (supervoxel->centroid_, adjacent_supervoxel_centers, ss.str (), pclViewer3);
        //Move iterator forward to next label
        label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
    }




}


void superVoxels_clustering(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud){

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
    pclViewer3->addPointCloud (colored_cloud, "colored voxels");
    pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "colored voxels");

    // Voxel Graph
    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);

    // Initialize a map with supervoxel indices and their normals
    std::map<uint32_t, pcl::PointNormal> supervoxel_normals;
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator it=supervoxel_clusters.begin(); it!=supervoxel_clusters.end(); ++it){


    }





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
    //    mls2.setUpsamplingRadius (0.05);
    //    mls2.setUpsamplingStepSize (0.005);

//    mls2.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZRGB, PointT>::VOXEL_GRID_DILATION);
//    mls2.setDilationVoxelSize(0.005);
//    mls2.setDilationIterations(1);

//     mls2.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZRGB, PointT>::RANDOM_UNIFORM_DENSITY);
//     mls2.setPointDensity(100);

    mls2.process (*cloud_smoothed2);

    //cloud_smoothed2 = downsample(cloud_smoothed2, 0.01);


//    // For VOXEL_GRID_DILATION only
//    pcl::PointCloud<PointT>::Ptr cloud_smoothed3(new pcl::PointCloud<PointT>);
//    for(int i=0; i<cloud_smoothed2->size(); i++){
//        PointT pt = cloud_smoothed2->at(i);
//        if(pt.x > -5 && pt.x < 5 && pt.y > -5 && pt.y < 5 && pt.z > -5 && pt.z < 5){
//        std::cout << "X : " << cloud_smoothed2->at(i).x << std::endl;
//        std::cout << "Y : " << cloud_smoothed2->at(i).y << std::endl;
//        std::cout << "Z : " << cloud_smoothed2->at(i).z << std::endl;
//            cloud_smoothed3->push_back(cloud_smoothed2->at(i));
//        }
//    }
//    cloud_smoothed2 = cloud_smoothed3;


    pcl::PointCloud<PointT>::Ptr cloud_smoothed3_translated(new pcl::PointCloud<PointT>);
    Eigen::Matrix4f cloud_translation;
    cloud_translation.setIdentity();
    cloud_translation(0,3) = 1; //x translation to compare with other mls
    pcl::transformPointCloud(*cloud_smoothed2, *cloud_smoothed3_translated, cloud_translation);



    pclViewer2->addPointCloud (cloud, ColorHandlerR(cloud, 0.0, 255.0, 0.0), "cloud");
    pclViewer2->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

    pclViewer2->addPointCloud (cloud_smoothed2, ColorHandlerT(cloud_smoothed2, 255.0, 255.0, 0.0), "smoothed2");
    pclViewer2->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "smoothed2");



    pcl::io::savePCDFile("cloud.pcd", *cloud);
    pcl::io::savePCDFile("cloud_smoothed.pcd", *cloud_smoothed2);

    return cloud_smoothed2;


}

void euclideanClusters(pcl::PointCloud<PointT>::Ptr cloud){

    pcl::ScopeTime t("Euclidean Clusters");

    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (0.04);
    ec.setMinClusterSize (20);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    pclViewer->removeAllPointClouds();
    pclViewer->addPointCloud (cloud, ColorHandlerT(cloud, 0.0, 255.0, 0.0), "scene_filtered");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene_filtered");

  //  std::cout << "Number of clusters : " << cluster_indices.at(0) << std::endl;

    for (int i=0; i < cluster_indices.size(); i++){
        pcl::PointIndices cloud_indices = cluster_indices.at(i);
        pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);

        for (int j=0; j<cloud_indices.indices.size(); j++){
            cloud_cluster->push_back (cloud->points[cloud_indices.indices[j]]);
        }
        cloud_cluster->height = 1;
        cloud_cluster->width = cloud_cluster->size();

        pcl::visualization::PointCloudColorHandlerRandom<PointT> randColor(cloud_cluster);
        std::stringstream ss;
        ss << i;
        std::string ind = ss.str();
        std::string pc_name = "object_" + ind;
        pclViewer->addPointCloud<PointT>(cloud_cluster, randColor, pc_name);
        pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, pc_name);

    }

}


void findCylinderPrimitive(pcl::PointCloud<PointT>::Ptr cloud){

    pcl::ScopeTime t("RANSAC CYLINDER");

    pcl::SACSegmentationFromNormals<PointT, PointT> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_CYLINDER);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight (0.1);
    seg.setMaxIterations (10000);
    seg.setDistanceThreshold (0.05);
    seg.setRadiusLimits (0.02, 0.1);
    seg.setInputCloud (cloud);
    seg.setInputNormals (cloud);

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    seg.segment (*inliers, *coefficients);
    std::cerr << "Cylinder coefficients: " << *coefficients << std::endl;
    std::cerr << "Cylinder inliers: " << *inliers << std::endl;

    pcl::PointCloud<PointT>::Ptr cylinder (new pcl::PointCloud<PointT> ());
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cylinder);

    if (cylinder->points.empty ()) std::cerr << "Can't find the cylindrical component." << std::endl;

    pclViewer3->addPointCloud (cylinder, ColorHandlerT(cylinder, 255.0, 255.0, 0.0), "Cylinder");
    pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Cylinder");

}

int main (int argc, char** argv){


    pcl::PointCloud<PointT>::Ptr scene_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr model_cloud(new pcl::PointCloud<PointT>);

    pcl::io::loadPCDFile("../bmw_clutter_remaining.pcd", *scene_cloud);
//    pcl::io::loadPCDFile("../customAlignment_fine.pcd", *model_cloud);

    initPCLViewer();

//    pcl::PointCloud<PointT>::Ptr scene_segmented(new pcl::PointCloud<PointT>);
//    scene_segmented = cropAndSegmentScene(scene_cloud, model_cloud);

//    // Smooth remaining points
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_segmented_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::PointCloud<PointT>::Ptr smoothed_cloud(new pcl::PointCloud<PointT>);
//    pcl::copyPointCloud(*scene_segmented, *scene_segmented_xyzrgb);
//    smoothed_cloud = smoothPointCloud(scene_segmented_xyzrgb);

   // pcl::io::savePCDFileASCII(std::string("../toSegment.pcd"), *smoothed_cloud);

   // boost::thread regionThread(region_growing_rgb_thread, smoothed_cloud);

   // regionGrowing(smoothed_cloud);
//    euclideanClusters(smoothed_cloud);
//  //  pclViewer->removeAllPointClouds();
//    pclViewer->addPointCloud (scene_cloud, ColorHandlerT(scene_cloud, 255.0, 0.0, 0.0), "scene");
//    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.9, "scene");


    // SuperVoxels
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::copyPointCloud(*scene_cloud, *scene_cloud_xyzrgb);
    superVoxels(scene_cloud_xyzrgb);


    // Find Primitives
//    findCylinderPrimitive(scene_segmented);



    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce (100);
        pclViewer2->spinOnce (100);
        pclViewer3->spinOnce (100);
    }

    return 0;
}
