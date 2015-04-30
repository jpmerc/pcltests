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
#include <pcl/filters/passthrough.h>

#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/crf_normal_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/features/normal_3d.h>

#include <boost/thread.hpp>
#include <vtkPolyLine.h>

#include "opencv2/imgproc/imgproc.hpp"

/*
  This file segments the input pointcloud into objects based on Learning to Manipulate Unknown Objects in Clutter by Reinforcement ( https://www.ri.cmu.edu/pub_files/2015/1/AbdeslamAAAI2015.pdf )
*/


typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> ColorHandlerR;
typedef pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> ColorHandlerRGB;

using namespace std;
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer2 (new pcl::visualization::PCLVisualizer ("3D Viewer2"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer3 (new pcl::visualization::PCLVisualizer ("3D Viewer3"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer4 (new pcl::visualization::PCLVisualizer ("3D Viewer4"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer5 (new pcl::visualization::PCLVisualizer ("3D Viewer5"));


// global matrices (not best implementation, but doing fast prototyping!!)
using namespace Eigen;
MatrixXd DegreesMatrix;
MatrixXd AdjacencyMatrix;



/// LIST OF PARAMETERS
double FACETS_SIMILARITY_THRESHOLD = 0.99;

double SUPERVOXEL_VOXEL_RESOLUTION = 0.01;
double SUPERVOXEL_SEED_RESOLUTION = 0.015;
double SUPERVOXEL_COLOR_IMPORTANCE = 0.1;
double SUPERVOXEL_SPATIAL_IMPORTANCE = 0.1;
double SUPERVOXEL_NORMAL_IMPORTANCE = 1.0;
bool SUPERVOXEL_REFINE = false;
int SUPERVOXEL_REFINE_ITERATIONS = 1;

double MEAN_SHIFT_ALPHA = 10.0;
int MEAN_SHIFT_ITERATIONS = 2;

int NUMBER_OF_SAME_NEIGHBORS_TO_MERGE_THRESHOLD = 2;

double KMEANS_EIGENVALUE_THRESHOLD = 0.9997;
int KMEANS_NUMBER_OF_CLUSTERS = 8;



double _distanceThreshold = 3.0;
double _pointColorThreshold = 3.0;
double _regionColorThreshold = 3.0;
double _minClusterSize = 1000;

void region_growing_rgb(pcl::PointCloud<PointT>::Ptr cloud);
void printSuperVoxels(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *voxels, std::multimap<int, int> *adjacency, boost::shared_ptr<pcl::visualization::PCLVisualizer> visual, bool showGraph);
void printMap(std::map<int, std::vector<int> > *myMap, std::string mapName);
void spectralClustering(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *voxels, std::multimap<int, int> *adjacency);
Eigen::VectorXd kmeans_clustering(Eigen::MatrixXd normalizedLaplacianMatrix);
void printEigenMatrix(Eigen::MatrixXd matrix, std::string name);
double numberOfSameNeighbors(std::multimap<int, int> *adjacency, int id1, int id2);
void mergeSimilarSupervoxelNeighbors(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *supervoxels, std::multimap<int, int> *adjacency);

std::map<int, std::vector<int> > findNeighbors(std::map<int, std::vector<int> > *adjacency,
                                               std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *supervoxels,
                                               std::multimap<int, int> *adjacencyNewClusters,
                                               std::multimap<int, int> *previousAdjacency);

Eigen::VectorXd cutGraph(Eigen::MatrixXd normalizedLaplacianMatrix, std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *voxels, std::multimap<int, int> *adjacency);

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

    pclViewer4->setBackgroundColor (0, 0, 0);
    pclViewer4->initCameraParameters ();
    pclViewer4->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow4 = pclViewer4->getRenderWindow();
    renderWindow4->SetSize(800,450);
    renderWindow4->Render();

    pclViewer5->setBackgroundColor (0, 0, 0);
    pclViewer5->initCameraParameters ();
    pclViewer5->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow5 = pclViewer5->getRenderWindow();
    renderWindow5->SetSize(800,450);
    renderWindow5->Render();

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


pcl::PointCloud<PointT>::Ptr extractPlane(pcl::PointCloud<PointT>::Ptr cloud, bool removePlane){

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
    extract.setNegative(removePlane);
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

    pcl::PointCloud<PointT>::Ptr bottom_plane = extractPlane(cloud,false);

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


double cosine_similarity(pcl::PointNormal *n1, pcl::PointNormal *n2){
    // Same orientation = 1
    // Orthogonal = 0
    // Opposite = -1

    double dot_product = (n1->normal_x * n2->normal_x) + (n1->normal_y * n2->normal_y) + (n1->normal_z * n2->normal_z);
    double n1_norm = sqrt( (n1->normal_x * n1->normal_x) + (n1->normal_y * n1->normal_y) + (n1->normal_z * n1->normal_z) );
    double n2_norm = sqrt( (n2->normal_x * n2->normal_x) + (n2->normal_y * n2->normal_y) + (n2->normal_z * n2->normal_z) );

    double similarity = dot_product / (n1_norm * n2_norm);


    return similarity;
}


void addPairToMultimapWithoutDoublons(std::multimap<int,int> *map, int id1, int id2, bool addReverse){

    bool foundPair = false;

    if(map->count(id1) > 0){
        std::multimap<int,int>::iterator  adjacent_itr = map->equal_range(id1).first;
        for ( ; adjacent_itr!=map->equal_range(id1).second; ++adjacent_itr)
        {
            int neighbor_index = adjacent_itr->second;
            if(neighbor_index == id2){
                foundPair = true;
                break;
            }
        }
    }

    if ((map->count(id2) > 0) && !foundPair){
        std::multimap<int,int>::iterator  adjacent_itr = map->equal_range(id2).first;
        for ( ; adjacent_itr!=map->equal_range(id2).second; ++adjacent_itr)
        {
            int neighbor_index = adjacent_itr->second;
            if(neighbor_index == id1){
                foundPair = true;
                break;
            }
        }

    }


    if(!foundPair){
        map->insert(std::pair<int,int>(id1, id2));
        if(addReverse){
            map->insert(std::pair<int,int>(id2, id1));
        }
    }
}


void addNeighbors(std::map<int, std::vector<int> > *adjacency,
                  std::map<int, std::vector<int> > *new_clusters,
                  std::map<uint32_t, pcl::PointNormal> *supervoxel_normals,
                  std::multimap<int,int> *adjacencyOldClusters,
                  std::vector<int> *processed_indices,
                  int index,
                  int cluster_index){

    if(adjacency->count(index) > 0){
        std::vector<int> neighbors_indices = adjacency->at(index);
        pcl::PointNormal n_1 = supervoxel_normals->at(index);

        for(int i = 0; i < neighbors_indices.size(); i++){

            // Chek if they are similar (if yes add it and check its neighbors)
            int neighbor_index = neighbors_indices.at(i);

            // Check if the point has already been processed to avoid being stuck in infinite loop
            if (std::find(processed_indices->begin(), processed_indices->end(), neighbor_index) == processed_indices->end()){

                pcl::PointNormal n_2 = supervoxel_normals->at(neighbor_index);
                double similarity = cosine_similarity(&n_1, &n_2);
                if(similarity >= FACETS_SIMILARITY_THRESHOLD){
                    processed_indices->push_back(neighbor_index);

                    // Check if the index is already present in the vector
                    std::vector<int> indices = new_clusters->at(cluster_index);
                    if(std::find(indices.begin(), indices.end(), neighbor_index) == indices.end()) {
                        //not present
                        indices.push_back(neighbor_index);
                        new_clusters->at(cluster_index) = indices;
                    }

                    addNeighbors(adjacency, new_clusters, supervoxel_normals, adjacencyOldClusters, processed_indices, neighbor_index, cluster_index);
                }

                else{
                    // Not similar (adjacent)
                    addPairToMultimapWithoutDoublons(adjacencyOldClusters, index, neighbor_index, false);
                }
            }

        }
    }


}

std::map<int, std::vector<int> > findNeighbors(std::map<int, std::vector<int> > *adjacency,
                                               std::map<uint32_t, pcl::PointNormal> *supervoxel_normals,
                                               std::multimap<int, int> *adjacencyNewClusters){

    int new_cluster_index = 0;
    std::map<int, std::vector<int> > new_clusters;
    std::multimap<int,int> adjacencyOldClusters;

    for(std::map<uint32_t, pcl::PointNormal>::iterator it = supervoxel_normals->begin(); it != supervoxel_normals->end(); ++it){
        int first = it->first;
        if(new_clusters.size() == 0){
            std::vector<int> v;
            v.push_back(first);
            new_clusters[new_cluster_index] = v;
            addNeighbors(adjacency, &new_clusters, supervoxel_normals, &adjacencyOldClusters, &v, first, new_cluster_index);
            new_cluster_index++;
        }

        else{
            bool found = false;
            for(std::map<int, std::vector<int> >::iterator it2 = new_clusters.begin(); it2 != new_clusters.end(); ++it2){
                std::vector<int> indices = it2->second;
                if(std::find(indices.begin(), indices.end(), first) != indices.end()) {
                    found = true;
                    break;
                }
            }

            if(!found){
                std::vector<int> v;
                v.push_back(first);
                new_clusters[new_cluster_index] = v;
                addNeighbors(adjacency, &new_clusters, supervoxel_normals, &adjacencyOldClusters, &v, first, new_cluster_index);
                new_cluster_index++;
            }
        }
    }


    // Fill the new Ajacency Matrix with the new cluster indices instead of the old ones
    //cout << "old adjacency : " << endl;
    for (std::multimap<int, int>::iterator it=adjacencyOldClusters.begin(); it!=adjacencyOldClusters.end(); ++it){
        //cout << it->first << " " << it->second << endl;
        int id1 = it->first;
        int id2 = it->second;
        int new_id1 = -1;
        int new_id2 = -1;
        bool found_id1 = false;
        bool found_id2 = false;

        std::map<int, std::vector<int> >::iterator it2;
        for(it2 = new_clusters.begin(); it2 != new_clusters.end(); ++it2){
            std::vector<int> indices = it2->second;
            if(!found_id1){
                if(std::find(indices.begin(), indices.end(), id1) != indices.end()) {
                    new_id1 = it2->first;
                    found_id1 = true;
                }
            }
            if(!found_id2){
                if(std::find(indices.begin(), indices.end(), id2) != indices.end()) {
                    new_id2 = it2->first;
                    found_id2 = true;
                }
            }
            if(found_id1 && found_id2) break;
        }

        if(new_id1 != new_id2) addPairToMultimapWithoutDoublons(adjacencyNewClusters, new_id1, new_id2, true);

    }

    return new_clusters;
}


void addNeighbors(std::map<int, std::vector<int> > *adjacency,
                  std::map<int, std::vector<int> > *new_clusters,
                  std::multimap<int,int> *adjacencyOldClusters,
                  std::multimap<int,int> *previousAdjacency,
                  std::vector<int> *processed_indices,
                  int index,
                  int cluster_index){

    if(adjacency->count(index) > 0){
        std::vector<int> neighbors_indices = adjacency->at(index);

        for(int i = 0; i < neighbors_indices.size(); i++){

            // Chek if they are similar (if yes add it and check its neighbors)
            int neighbor_index = neighbors_indices.at(i);

            // Check if the point has already been processed to avoid being stuck in infinite loop
            if (std::find(processed_indices->begin(), processed_indices->end(), neighbor_index) == processed_indices->end()){

                double similar_neighbors = numberOfSameNeighbors(previousAdjacency, index, neighbor_index);
                if(similar_neighbors >= NUMBER_OF_SAME_NEIGHBORS_TO_MERGE_THRESHOLD){
                    processed_indices->push_back(neighbor_index);

                    // Check if the index is already present in the vector
                    std::vector<int> indices = new_clusters->at(cluster_index);
                    if(std::find(indices.begin(), indices.end(), neighbor_index) == indices.end()) {
                        //not present
                        indices.push_back(neighbor_index);
                        new_clusters->at(cluster_index) = indices;
                    }

                    addNeighbors(adjacency, new_clusters, adjacencyOldClusters, previousAdjacency, processed_indices, neighbor_index, cluster_index);
                }

                else{
                    // Not similar (adjacent)
                    addPairToMultimapWithoutDoublons(adjacencyOldClusters, index, neighbor_index, false);
                }
            }

        }
    }


}

std::map<int, std::vector<int> > findNeighbors(std::map<int, std::vector<int> > *adjacency,
                                               std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *supervoxels,
                                               std::multimap<int, int> *adjacencyNewClusters,
                                               std::multimap<int, int> *previousAdjacency){

    int new_cluster_index = 0;
    std::map<int, std::vector<int> > new_clusters;
    std::multimap<int,int> adjacencyOldClusters;

    for(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr>::iterator it = supervoxels->begin(); it != supervoxels->end(); ++it){
        int first = it->first;
        if(new_clusters.size() == 0){
            std::vector<int> v;
            v.push_back(first);
            new_clusters[new_cluster_index] = v;
            addNeighbors(adjacency, &new_clusters, &adjacencyOldClusters, previousAdjacency, &v, first, new_cluster_index);
            new_cluster_index++;
        }

        else{
            bool found = false;
            for(std::map<int, std::vector<int> >::iterator it2 = new_clusters.begin(); it2 != new_clusters.end(); ++it2){
                std::vector<int> indices = it2->second;
                if(std::find(indices.begin(), indices.end(), first) != indices.end()) {
                    found = true;
                    break;
                }
            }

            if(!found){
                std::vector<int> v;
                v.push_back(first);
                new_clusters[new_cluster_index] = v;
                addNeighbors(adjacency, &new_clusters, &adjacencyOldClusters, previousAdjacency, &v, first, new_cluster_index);
                new_cluster_index++;
            }
        }
    }


    // Fill the new Ajacency Matrix with the new cluster indices instead of the old ones
    //cout << "old adjacency : " << endl;
    for (std::multimap<int, int>::iterator it=adjacencyOldClusters.begin(); it!=adjacencyOldClusters.end(); ++it){
        //cout << it->first << " " << it->second << endl;
        int id1 = it->first;
        int id2 = it->second;
        int new_id1 = -1;
        int new_id2 = -1;
        bool found_id1 = false;
        bool found_id2 = false;

        std::map<int, std::vector<int> >::iterator it2;
        for(it2 = new_clusters.begin(); it2 != new_clusters.end(); ++it2){
            std::vector<int> indices = it2->second;
            if(!found_id1){
                if(std::find(indices.begin(), indices.end(), id1) != indices.end()) {
                    new_id1 = it2->first;
                    found_id1 = true;
                }
            }
            if(!found_id2){
                if(std::find(indices.begin(), indices.end(), id2) != indices.end()) {
                    new_id2 = it2->first;
                    found_id2 = true;
                }
            }
            if(found_id1 && found_id2) break;
        }

        if(new_id1 != new_id2) addPairToMultimapWithoutDoublons(adjacencyNewClusters, new_id1, new_id2, true);

    }

    return new_clusters;
}

template<typename T,typename K> std::map<T, std::vector<T> > convertMultimapToMap(std::multimap<K, K> *supervoxel_adjacency){
    int id = -1;
    std::vector<T> n_id;
    std::map<T, std::vector<T> > adjacency_map;
    typename std::multimap<K, K>::iterator it;
    for( it = supervoxel_adjacency->begin(); it != supervoxel_adjacency->end(); ){
        //cout << it->first << " , " << it->second << endl;

        if(id < 0) id = it->first;

        if(id != it->first){
            adjacency_map[id] = n_id;
            n_id.clear();
            id = it->first;
        }
        n_id.push_back(it->second);


        ++it;
        if(it == supervoxel_adjacency->end()) {
            adjacency_map[id] = n_id;
        }
    }

    return adjacency_map;
}


template<typename T,typename K> std::map <int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> mergeSuperVoxels(std::map<T, std::vector<T> > *new_clusters_map,
                                                                                                         std::map <K, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr > *supervoxel_clusters){

    typedef pcl::PointXYZRGBA PointT;

    std::map <int, pcl::Supervoxel<PointT>::Ptr > merged_supervoxel_clusters;

    for(typename std::map<T, std::vector<T> >::iterator it = new_clusters_map->begin(); it != new_clusters_map->end(); ++it){
        int key = it->first;
        std::vector<int> v = it->second;

        pcl::Supervoxel<PointT>::Ptr merged_sVoxel(new pcl::Supervoxel<PointT>);

        for(int i=0; i < v.size(); i++){
            int id = v.at(i);
            pcl::Supervoxel<PointT>::Ptr sVoxel = supervoxel_clusters->at(id);
            *(merged_sVoxel->voxels_)  += *(sVoxel->voxels_);
            *(merged_sVoxel->normals_) += *(sVoxel->normals_);
        }

        // Set Centroid of the merged Supervoxel
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid<PointT>(*(merged_sVoxel->voxels_), centroid);
        pcl::PointXYZRGBA cent;
        cent.x = centroid[0];
        cent.y = centroid[1];
        cent.z = centroid[2];
        merged_sVoxel->centroid_ = cent;

        // Set Normal of the merged Supervoxel
        Eigen::Vector4f vNormal = Eigen::Vector4f::Zero();
        float curvature = 0;

        /// Problem (can return Nan)
        pcl::PointCloud<PointT> voxels_pc = *(merged_sVoxel->voxels_);
        pcl::computePointNormal(voxels_pc, vNormal, curvature);
        vNormal[3] = 0.0;
        vNormal.normalize();

        // If there is less than 2 points in the new cluster, take the average of the supervoxel centroid normals
        if(isnan(vNormal[0]) || isnan(vNormal[1]) || isnan(vNormal[2])){
            Eigen::Vector4f tempNormal = Eigen::Vector4f::Zero();
            for(int i=0; i < v.size(); i++){
                int id = v.at(i);
                pcl::Supervoxel<PointT>::Ptr sVoxel = supervoxel_clusters->at(id);
                tempNormal += sVoxel->normal_.getNormalVector4fMap();
            }
            vNormal = tempNormal / v.size();
            vNormal.normalize();
        }

        pcl::Normal norm;
        norm.normal_x = vNormal[0];
        norm.normal_y = vNormal[1];
        norm.normal_z = vNormal[2];
        norm.curvature = curvature;
        merged_sVoxel->normal_ = norm;

        merged_supervoxel_clusters[key] = merged_sVoxel;
    }

    return merged_supervoxel_clusters;
}


void superVoxels_clustering(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud){

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;
    typedef pcl::PointNormal PointNT;
    typedef pcl::PointCloud<PointNT> PointNCloudT;
    typedef pcl::PointXYZL PointLT;
    typedef pcl::PointCloud<PointLT> PointLCloudT;

    pcl::ScopeTime t("Segmentation in Supervoxels");

    bool use_transform = false;

    /// Supervoxel Clustering
    pcl::SupervoxelClustering<PointT> super (SUPERVOXEL_VOXEL_RESOLUTION, SUPERVOXEL_SEED_RESOLUTION, use_transform);
    super.setInputCloud (cloud);
    super.setColorImportance (SUPERVOXEL_COLOR_IMPORTANCE);
    super.setSpatialImportance (SUPERVOXEL_SPATIAL_IMPORTANCE);
    super.setNormalImportance (SUPERVOXEL_NORMAL_IMPORTANCE);

    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    super.extract (supervoxel_clusters);

    if(SUPERVOXEL_REFINE) super.refineSupervoxels(SUPERVOXEL_REFINE_ITERATIONS, supervoxel_clusters);
    //super.refineSupervoxels(3, supervoxel_clusters);
    pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());


    // copy map to good format for spectral clustering (int instead of uint32_t)
    std::map <int, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters_int;
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator it=supervoxel_clusters.begin(); it!=supervoxel_clusters.end(); ++it){
        int id = it->first;
        pcl::Supervoxel<PointT>::Ptr vox = it->second;
        supervoxel_clusters_int[id] = vox;
    }


    // Voxel Cloud Colors
    PointCloudT::Ptr colored_cloud = super.getColoredVoxelCloud ();
    cout << "Colored Cloud Size : " << colored_cloud->size() << endl;
    pclViewer3->addPointCloud (colored_cloud, "colored voxels");
    pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "colored voxels");

    // Voxel Graph
    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);

    // copy map to good format for spectral clustering (int instead of uint32_t)
    std::multimap<int, int> supervoxel_adjacency_int;
    for (std::multimap<uint32_t, uint32_t>::iterator it=supervoxel_adjacency.begin(); it!=supervoxel_adjacency.end(); ++it){
        supervoxel_adjacency_int.insert(std::pair<int,int>(it->first, it->second));
    }


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



    // Initialize a map with supervoxel indices and their normals
    std::map<uint32_t, pcl::PointNormal> supervoxel_normals;
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator it=supervoxel_clusters.begin(); it!=supervoxel_clusters.end(); ++it){
        pcl::PointNormal ptN;
        it->second->getCentroidPointNormal(ptN);
        //cout << it->first << " , " << ptN << endl;
        //cout << it->first << endl;
        supervoxel_normals[it->first] = ptN;
    }



    // DEBUG
    int size = 0;
   //cout << "[" ;
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator it=supervoxel_clusters.begin(); it!=supervoxel_clusters.end(); ++it){
        size += it->second->voxels_->size();
        //cout << it->first << " " << it->second->voxels_->size() << "; ";
    }
   // cout << "]" << endl;
   // cout << "Total Number of points : " << size << endl;


    // Change the format of the adjacency map for easier calculation of mean shift
    std::map<int, std::vector<int> > adjacency_map = convertMultimapToMap<int,uint32_t>(&supervoxel_adjacency);
    printMap(&adjacency_map, "Adjacency Map");


    /// Starting Mean Shift Algorithm
    // Mean shift iterations for calculating new normals from neighborhood
    for(int mean_shift_iterations = 0; mean_shift_iterations < MEAN_SHIFT_ITERATIONS; mean_shift_iterations++){
        // Loop on supervoxels
        for(std::map<int, std::vector<int> >::iterator it = adjacency_map.begin(); it != adjacency_map.end(); ++it){
            std::vector<int> indices = it->second;

            double eta = 0;
            pcl::PointNormal n_i = supervoxel_normals[it->first];
            pcl::PointNormal calculated_normal = n_i;
            calculated_normal.normal_x = 0;
            calculated_normal.normal_y = 0;
            calculated_normal.normal_z = 0;

            //cout << n_i << endl;
            for(int i=0; i < indices.size(); i++){
                pcl::PointNormal n_j = supervoxel_normals[indices.at(i)];
                // cout << n_j << endl;
                double dot_product = (n_i.normal_x * n_j.normal_x) + (n_i.normal_y * n_j.normal_y) + (n_i.normal_z * n_j.normal_z);
                //cout << "Dot : " << dot_product << endl;
                double value = 0;
                if(dot_product >= 1.0){
                    value = 1;
                }
                else {
                    value = exp( -MEAN_SHIFT_ALPHA * acos(dot_product) );
                }
                // cout << "Value : " << value << endl;
                calculated_normal.normal_x += (value * n_j.normal_x);
                calculated_normal.normal_y += (value * n_j.normal_y);
                calculated_normal.normal_z += (value * n_j.normal_z);
                eta += value;
                //cout << "Similarity (" << it->first << ", " << indices.at(i) << ") = " << cosine_similarity(&n_i, &n_j) << std::endl;
            }

            calculated_normal.normal_x = (1/eta) * calculated_normal.normal_x;
            calculated_normal.normal_y = (1/eta) * calculated_normal.normal_y;
            calculated_normal.normal_z = (1/eta) * calculated_normal.normal_z;

            supervoxel_normals[it->first] = calculated_normal;

            // cout << "Eta : " << eta << endl;
            // cout << "New Normal : " << calculated_normal << endl;
        }

    }

    /// Merge Supervoxels Based on Cosine Similarity
    std::map<int, std::vector<int> > new_clusters_map;
    std::multimap<int, int> new_adj_map;
    new_clusters_map = findNeighbors(&adjacency_map, &supervoxel_normals, &new_adj_map);


    cout << "------------- New Adjacency Map-------------" << endl;
    for(std::multimap<int, int>::iterator it = new_adj_map.begin(); it != new_adj_map.end(); ++it){
        int key = it->first;
        int value = it->second;

        cout << "(" << key << ", " << value << ")" << endl;
    }


    printMap(&new_clusters_map, "New Clusters Map");


    /// Build a map of the newly merged Supervoxels
    cout << "Merging...!" << endl;
    std::map <int, pcl::Supervoxel<PointT>::Ptr > merged_supervoxel_clusters = mergeSuperVoxels<int,uint32_t>(&new_clusters_map, &supervoxel_clusters);


    // Print Centroid Matrix
    cout << "Centroid_Matrix = [ " << endl;
    for(std::map <int, pcl::Supervoxel<PointT>::Ptr >::iterator it = merged_supervoxel_clusters.begin(); it != merged_supervoxel_clusters.end(); ++it){
        int key = it->first;
        pcl::Supervoxel<PointT>::Ptr supervox = it->second;
        cout << supervox->centroid_.x << " ";
        cout << supervox->centroid_.y << " ";
        cout << supervox->centroid_.z << ";" << endl;
    }
    cout << "]" << endl;

    // Print Normals Matrix
    cout << "Normals_Matrix = [ " << endl;
    for(std::map <int, pcl::Supervoxel<PointT>::Ptr >::iterator it = merged_supervoxel_clusters.begin(); it != merged_supervoxel_clusters.end(); ++it){
        int key = it->first;
        pcl::Supervoxel<PointT>::Ptr supervox = it->second;
        cout << supervox->normal_.normal_x << " ";
        cout << supervox->normal_.normal_y << " ";
        cout << supervox->normal_.normal_z << ";" << endl;
    }
    cout << "]" << endl;


    printSuperVoxels(&merged_supervoxel_clusters, &new_adj_map, pclViewer2, true);


    // Merge facets with same neighbors
    //mergeSimilarSupervoxelNeighbors(&merged_supervoxel_clusters, &new_adj_map);
    //printSuperVoxels(&merged_supervoxel_clusters, &new_adj_map, pclViewer5, true);

    spectralClustering(&merged_supervoxel_clusters, &new_adj_map);

    //spectralClustering(&supervoxel_clusters_int, &supervoxel_adjacency_int);



}

void mergeSimilarSupervoxelNeighbors(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *supervoxels, std::multimap<int, int> *adjacency){

    typedef pcl::PointXYZRGBA PointT;

    std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> merged_supervoxels;

    std::map<int, std::vector<int> > adjacency_map = convertMultimapToMap<int,int>(adjacency);

    std::multimap<int, int> adjacencyNewClusters;
    std::map<int, std::vector<int> > merged_indices = findNeighbors(&adjacency_map, supervoxels, &adjacencyNewClusters, adjacency);

    printMap(&merged_indices, "Similar Neighbors Merging");

    std::map <int, pcl::Supervoxel<PointT>::Ptr > merged_supervoxel_clusters = mergeSuperVoxels<int,int>(&merged_indices, supervoxels);

    supervoxels->swap(merged_supervoxel_clusters);
    adjacency->swap(adjacencyNewClusters);
}

bool isAdjacent(std::multimap<int, int> *adjacency, int id1, int id2){

    bool adj = false;
    std::multimap<int,int>::iterator it;
    for (it = adjacency->equal_range(id1).first ; it!=adjacency->equal_range(id1).second; ++it){
        if(it->second == id2){
            adj = true;
            break;
        }
    }
    return adj;
}

double numberOfSameNeighbors(std::multimap<int, int> *adjacency, int id1, int id2){

    std::vector<int> v1;
    std::vector<int> v2;

    std::multimap<int,int>::iterator it;
    for (it = adjacency->equal_range(id1).first ; it!=adjacency->equal_range(id1).second; ++it){
        int temp_id = it->second;
        v1.push_back(temp_id);
    }

    for (it = adjacency->equal_range(id2).first ; it!=adjacency->equal_range(id2).second; ++it){
        int temp_id = it->second;
        v2.push_back(temp_id);
    }

    double sameNeighbors = 0;
    for(int i = 0; i < v1.size(); i++){
        int temp_id_1 = v1.at(i);
        for(int j = 0; j < v2.size(); j++){
           int temp_id_2 = v2.at(j);
           if(temp_id_1 == temp_id_2){
               sameNeighbors++;
               continue;
           }
        }

    }

    return sameNeighbors;
}




void spectralClustering(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *voxels, std::multimap<int, int> *adjacency){

    using namespace Eigen;

    int size = voxels->size();


    // Init Normalized Laplacian Matrix
    MatrixXd normalizedLaplacian;
    normalizedLaplacian.resize(size, size);
    normalizedLaplacian.setZero();


    // Init weight (Adjacency)  Matrix
    MatrixXd W;
    W.resize(size, size);
    W.setZero();

    // Init degree Matrix
    MatrixXd D;
    D.resize(size, size);
    D.setZero();

    // Init identity Matrix
    MatrixXd I;
    I.resize(size, size);
    I.setIdentity();

    // Number Of Connections
    MatrixXd Connections;
    Connections.resize(size, size);
    Connections.setZero();

    int i = 0;
    int j = 0;

    std::map<int,int> clusterToMatrixIndices;

    // Fill Weight (adjacency matrix)
    for(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr>::iterator it = voxels->begin(); it != voxels->end(); ++it){

        int id1 = it->first;
        pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr supervoxel1 = it->second;
        j = 0;
        clusterToMatrixIndices[id1] = i;

        //cout << id1 << endl;
        for(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr>::iterator it2 = voxels->begin(); it2 != voxels->end(); ++it2){
            int id2 = it2->first;
            pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr supervoxel2 = it2->second;

            // Abdeslam Method Weights
            //            // Weight Matrix
            //            pcl::Normal v1 = supervoxel1->normal_;
            //            pcl::PointXYZRGBA c1 = supervoxel1->centroid_;

            //            pcl::Normal v2 = supervoxel2->normal_;
            //            pcl::PointXYZRGBA c2 = supervoxel2->centroid_;

            //            double opt1 = v1.normal_x * (c1.x - c2.x) + v1.normal_y * (c1.y - c2.y) + v1.normal_z * (c1.z - c2.z);
            //            double opt2 = v2.normal_x * (c2.x - c1.x) + v2.normal_y * (c2.y - c1.y) + v2.normal_z * (c2.z - c1.z);

            //            double max = std::max(opt1, opt2);
            //            if(isnan(max) || max < 0) max = 0;

            // Alexandrov Weights
            //            const Vector3f& n1 = supervoxel1->normal_.getNormalVector3fMap().normalized();
            //            const Vector3f& c1 = supervoxel1->centroid_.getVector3fMap();

            //            const Vector3f& n2 = supervoxel2->normal_.getNormalVector3fMap().normalized();
            //            const Vector3f& c2 = supervoxel2->centroid_.getVector3fMap();

            //            double dot = (n2 - n1).dot(c2 - c1);
            //            if(dot > 0){
            //                // convex
            //                W(i, j) = 1;
            //            }
            //            else{
            //                // inverse quadratic radial basis function : 1 / ( 1 + sigma* ||(n1-n2)||^2)
            //                double sigma = 0.05;
            //                double squared_norm = (n1 - n2).squaredNorm();
            //                double function_value = 1 / (1 + sigma * squared_norm);
            //                W(i, j) = function_value;
            //            }

            if(isAdjacent(adjacency, id1, id2)){
                const Vector3f& c1 = supervoxel1->centroid_.getVector3fMap();
                const Vector3f& c2 = supervoxel2->centroid_.getVector3fMap();
                double sigma = 0.05;
                double squared_norm = (c1 - c2).squaredNorm();
                double val = exp(-squared_norm / (2*sigma*sigma));
                double inverse_quadratic = 1 / (1 + 500 * squared_norm);

//                double sameNeighbors = numberOfSameNeighbors(adjacency, id1, id2);
//                double neighborhood_score = 1.5 * log(sameNeighbors + 1);
//                if(neighborhood_score > 1.0) neighborhood_score = 1.0;

//                double score = ( inverse_quadratic + neighborhood_score ) / 2 ;

                W(i, j) = val;
            }
            else{
                 W(i, j) = 0;
            }

            j++;
        }
        i++;
    }



    // Fill Degrees Matrix (sum of weights of adjacents supervoxels)
    for(int x = 0; x < D.rows(); x++){

        int numberOfAdjacentSuperVoxels = adjacency->count(x);
        Connections(x,x) = numberOfAdjacentSuperVoxels;
        if(numberOfAdjacentSuperVoxels > 0){
            std::multimap<int,int>::iterator it;
            double sum_of_weights = 0;
            for (it = adjacency->equal_range(x).first ; it!=adjacency->equal_range(x).second; ++it){
                int first = it->first;
                int second = it->second;
                sum_of_weights += W(clusterToMatrixIndices[first], clusterToMatrixIndices[second]);
            }
            D(x,x) = sum_of_weights;
        }

        else{
            D(x,x) = 0;
        }
    }

    MatrixXd D_root = D.cwiseSqrt();
    MatrixXd D_root_inv = D_root;
    for(int i = 0; i < size; i++){
        double v = D_root_inv(i, i);
        if(v != 0) D_root_inv(i, i) = 1 / v;
    }

    // Calculate Normalized Laplacian Matrix
    MatrixXd Laplacian = D - W;
    normalizedLaplacian = D_root_inv * Laplacian * D_root_inv;

    // Random Walk Laplacian (replace zeros in D by epsilon)
    MatrixXd D_without_zeros;
    D_without_zeros.resize(size, size);
    D_without_zeros.setZero();
    double epsilon = 2.2204 * pow(10.0, -16.0);
    for(int x = 0; x < D.rows(); x++){
        if(D(x,x) == 0){
            D_without_zeros(x,x) = epsilon;
        }
        else{
            D_without_zeros(x,x) = D(x,x);
        }
    }
    MatrixXd RandomWalkLaplacian;
    MatrixXd D_inverse = D_without_zeros.inverse();
    RandomWalkLaplacian = D_inverse * Laplacian;

    // Print Matrices
    printEigenMatrix(W, "Adjacency (weight) Matrix");
    printEigenMatrix(D, "Degree Matrix");
    printEigenMatrix(Laplacian, "Laplacian Matrix");
    printEigenMatrix(normalizedLaplacian, "Normalized Laplacian Matrix");
    printEigenMatrix(RandomWalkLaplacian, "Random Walk Laplacian Matrix");
    printEigenMatrix(D_root_inv, "Root Degree inverse Matrix");
    printEigenMatrix(D_inverse, "Degree inverse Matrix");
    printEigenMatrix(Connections, "Connections Matrix (not used in code, only for information purposes)");

    DegreesMatrix = D;
    AdjacencyMatrix = W;



//    cout << "Degree Matrix root" << endl;
//    cout << D_root  << endl;

//    cout << "Degree Matrix root inversed" << endl;
//    cout << D_root_inv  << endl;

//    cout << "normalizedLaplacian Matrix" << endl;
//    cout << normalizedLaplacian << endl;


    VectorXd new_labels = kmeans_clustering(normalizedLaplacian);

    cutGraph(normalizedLaplacian, voxels, adjacency);


    std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> new_supervoxels;

    int ind = 0;
    for(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr>::iterator it = voxels->begin(); it != voxels->end(); ++it){
        pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr supervox = it->second;
        int new_id = new_labels(ind);
         if(new_supervoxels.count(new_id) > 0){
             pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr new_supervox = new_supervoxels[new_id];
             *(new_supervox->voxels_) += *(supervox->voxels_);
             new_supervoxels[new_id] = new_supervox;
         }

         else{
             new_supervoxels[new_id] = supervox;
         }

         ind++;
    }

    printSuperVoxels(&new_supervoxels, adjacency, pclViewer4, false);

}

cv::Mat retrieveKsmallestEigenvectors(Eigen::MatrixXd eigenvectors, Eigen::MatrixXd eigenvalues, int k){

    std::vector<int> eigen_indices;
    cv::Mat samples(eigenvectors.rows() , k, CV_32F, 0.0);

    // <eigenvalue, index_in_matrix>
    std::multimap<double,int> eigenvalues_map;

    int OriginalNumberOfClusters = eigenvalues.rows();
    for(int i = 0; i < OriginalNumberOfClusters; i++){
        double number = eigenvalues(i);
        eigenvalues_map.insert(std::pair<double,int>(number, i));
    }

    cout << "cv::orting :" << endl;
    int sorted_index = 0;
    for(std::multimap<double,int>::iterator it = eigenvalues_map.begin(); it != eigenvalues_map.end(); ++it){
        int old_index = it->second;
        cout << it->first << " , " <<  old_index << endl;
       // cout << eigenvectors.col(old_index) << endl;

        for(int i = 0; i < eigenvectors.rows(); i++){
            double number = eigenvectors(i, old_index) ;
            samples.at<float>(i, sorted_index) = number;
        }

        sorted_index ++;
        if(sorted_index >= k) break;

    }


    cout << samples << endl;

    return samples;
}

Eigen::MatrixXd sortEigenvectors(Eigen::MatrixXd eigenvectors, Eigen::MatrixXd eigenvalues){
    using namespace Eigen;

    MatrixXd sorted_eigenvectors;
    sorted_eigenvectors.resize(eigenvectors.rows(), eigenvectors.cols());
    sorted_eigenvectors.setZero();

    // <eigenvalue, index_in_matrix>
    std::multimap<double,int> eigenvalues_map;

    int OriginalNumberOfClusters = eigenvalues.rows();
    for(int i = 0; i < OriginalNumberOfClusters; i++){
        double number = eigenvalues(i);
        eigenvalues_map.insert(std::pair<double,int>(number, i));
    }

    cout << "Test sorting :" << endl;
    int numberOfCols = eigenvectors.cols();
    int sorted_index = 0;
    for(std::multimap<double,int>::iterator it = eigenvalues_map.begin(); it != eigenvalues_map.end(); ++it){
        int old_index = it->second;
        cout << it->first << " , " <<  old_index << endl;
       // cout << eigenvectors.col(old_index) << endl;

        for(int i = 0; i < eigenvectors.rows(); i++){
            double number = eigenvectors(i, old_index) ;
            sorted_eigenvectors(i, sorted_index) = number;
        }

        sorted_index ++;
        if(sorted_index >= numberOfCols) break;
    }

    return sorted_eigenvectors;
}

Eigen::MatrixXd complexToFloatMatrix(Eigen::MatrixXcd matrix){

    using namespace Eigen;
    MatrixXd float_matrix;
    float_matrix.resize(matrix.rows(),matrix.cols());

    for(int i = 0; i < matrix.rows(); i++){
        for(int j = 0; j < matrix.cols(); j++){
            std::complex<double> vec = matrix(i,j);
            double real = vec.real();
            double im = vec.imag();
            float_matrix(i,j) = real;
        }
    }

    return float_matrix;
}


int findElbowIndex(std::vector<double> *distances){


    /// Find the most distant point

    int vectorLastIndex = distances->size() - 1;
    // slope = (y2-y1)/(x2-x1)
    double slope = (distances->at(vectorLastIndex) - distances->at(0)) / (vectorLastIndex - 0);
    double intercept =  distances->at(0);

    double A = slope;
    double B = -1.0;
    double C = intercept;

    double largestDistance = 0;
    int largestDistanceIndex = -1;

    for(int i = 0; i < distances->size(); i++){

        double numerator = std::abs( A*i + B*distances->at(i) + C );
        double denominator = sqrt(A*A + B*B);
        double ptToLineDistance = numerator / denominator;

        if(ptToLineDistance >= largestDistance){
            // added a little offset to advantage more clusters (better to oversegment than undersegment in our cas)
            largestDistance = ptToLineDistance - 0.1;
            largestDistanceIndex = i;
        }

        cout << ptToLineDistance << endl;
    }

    return largestDistanceIndex;
}

void eigenDecomposition(Eigen::MatrixXd MatrixToDecompose, Eigen::MatrixXd &eigenvectors, Eigen::MatrixXd &eigenvalues){
    using namespace Eigen;

    EigenSolver<MatrixXd> es(MatrixToDecompose);

    MatrixXcd eigenvectors_matrix = es.eigenvectors();
    MatrixXcd eigenvalues_matrix = es.eigenvalues();

    cout << "Eigenvalues :" << endl;
    cout <<  es.eigenvalues() << endl;

    eigenvectors = complexToFloatMatrix(eigenvectors_matrix);
    printEigenMatrix(eigenvectors, "Eigenvectors");

    eigenvalues = complexToFloatMatrix(eigenvalues_matrix);
    printEigenMatrix(eigenvalues, "Eigenvalues");
}




// Adds neighbors (no condition)
void addConnectedComponents(std::map<int, std::vector<int> > *adjacency,
                  std::map<int, std::vector<int> > *new_clusters,
                  std::vector<int> *processed_indices,
                  int index,
                  int cluster_index){

    if(adjacency->count(index) > 0){
        std::vector<int> neighbors_indices = adjacency->at(index);

        for(int i = 0; i < neighbors_indices.size(); i++){

            int neighbor_index = neighbors_indices.at(i);

            // Check if the point has already been processed to avoid being stuck in infinite loop
            if (std::find(processed_indices->begin(), processed_indices->end(), neighbor_index) == processed_indices->end()){

                processed_indices->push_back(neighbor_index);

                // Check if the index is already present in the vector
                std::vector<int> indices = new_clusters->at(cluster_index);
                if(std::find(indices.begin(), indices.end(), neighbor_index) == indices.end()) {
                    //not present
                    indices.push_back(neighbor_index);
                    new_clusters->at(cluster_index) = indices;

                    addConnectedComponents(adjacency, new_clusters, processed_indices, neighbor_index, cluster_index);
                }
            }
        }
    }


}

// Find connected components
std::map<int, std::vector<int> > findConnectedComponents(std::map<int, std::vector<int> > *adjacency, std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *supervoxels)
{

    int new_cluster_index = 0;
    std::map<int, std::vector<int> > new_clusters;

    for(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr>::iterator it = supervoxels->begin(); it != supervoxels->end(); ++it){
        int first = it->first;
        if(new_clusters.size() == 0){
            std::vector<int> v;
            v.push_back(first);
            new_clusters[new_cluster_index] = v;
            addConnectedComponents(adjacency, &new_clusters, &v, first, new_cluster_index);
            new_cluster_index++;
        }

        else{
            bool found = false;
            for(std::map<int, std::vector<int> >::iterator it2 = new_clusters.begin(); it2 != new_clusters.end(); ++it2){
                std::vector<int> indices = it2->second;
                if(std::find(indices.begin(), indices.end(), first) != indices.end()) {
                    found = true;
                    break;
                }
            }

            if(!found){
                std::vector<int> v;
                v.push_back(first);
                new_clusters[new_cluster_index] = v;
                addConnectedComponents(adjacency, &new_clusters, &v, first, new_cluster_index);
                new_cluster_index++;
            }
        }
    }



    return new_clusters;
}



cv::Mat eigenToCVMat(Eigen::MatrixXd matrix){

    cv::Mat cv_mat(matrix.rows() , matrix.cols(), CV_32F, 0.0);

    for(int i = 0; i < matrix.rows(); i++){
        for(int j = 0; j < matrix.cols(); j++){
            double number = matrix(i, j) ;
            cv_mat.at<float>(i, j) = number;
        }
    }

    return cv_mat;
}

void updateConnectivityMatrix(Eigen::MatrixXd &connectivity, std::multimap<int,int> *label_map){

    int index = 0;
    std::vector<int> indices;
    for(std::multimap<int,int>::iterator it = label_map->begin(); it != label_map->end(); ++it){
        int first = it->first;
        int second = it->second;
        cout <<  first << " .. " << second << endl;

        if(first == index){
            if(!indices.empty()){
                for(int i = 0; i < indices.size(); i++){
                    int id = indices.at(i);
                    connectivity(id, second) = connectivity(id, second) + 1;
                    connectivity(second, id) = connectivity(second, id) + 1;
                }

            }
            indices.push_back(second);

        }

        else{
            index++;
            indices.clear();
            indices.push_back(second);
        }
    }

}

void updateCutHypotheses(Eigen::MatrixXd &connectivity, std::multimap<int,int> *adjacency, std::multimap<int,int> *hypotheses){


    for(std::multimap<int,int>::iterator it = adjacency->begin(); it != adjacency->end(); ++it){
        int first = it->first;
        int second = it->second;
        int numberOfTimesInSameCluster = connectivity(first,second);
        if(numberOfTimesInSameCluster <= 2){
            // Have been separated at least one time (for k = 2..4 clusters)
            addPairToMultimapWithoutDoublons(hypotheses, first, second, false);
        }
    }
}


std::multimap<int,int> getGraphCutHypotheses(Eigen::MatrixXd eigenVectors, std::multimap<int, int> *adjacency){

    using namespace Eigen;

    std::multimap<int,int> hypotheses;

    // Loop over the eigenvectors to discover cut hypotheses and test NCut on hypotheses
    //for(int i = 0; i < sorted_vectors; i++){
    for(int i = 0; i < 10; i++){
        // Column i
        //std::map<double, int> ordered_data; //<data,index>
        MatrixXd eigen_column = eigenVectors.col(i);
        cv::Mat cv_eigen_col = eigenToCVMat(eigen_column);

        MatrixXd connectivity;
        connectivity.resize(eigenVectors.rows(),eigenVectors.rows());
        connectivity.setZero();

        cv::Mat labels;
        int attempts = 5;
        cv::Mat centers;

        cout << "kmeans_linear" << endl;
        for(int k = 2; k < 5; k++){
            double distance = cv::kmeans(cv_eigen_col, k, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.00001), attempts, cv::KMEANS_PP_CENTERS, centers );
           // cout << k << " " << distance << ";" << endl;

            // sort the data <cluster_number, cluster_index>
            std::multimap<int,int> label_map;
            for(int j = 0; j < labels.rows; j++){
                int cluster_number = labels.at<int>(j);
                label_map.insert(std::pair<int,int>(cluster_number, j));
            }

            // Increase the connectivity of two clusters if they are labeled the same by kmeans
            updateConnectivityMatrix(connectivity, &label_map);
            cout << connectivity << endl;
        }

        // Now check the scores for the adjacent ones (if they are separated at least once by Kmeans, add it to cut hypotheses)
        updateCutHypotheses(connectivity, adjacency, &hypotheses);
        cout << "Hypotheses : " << endl;
        for(std::multimap<int,int>::iterator it = hypotheses.begin(); it != hypotheses.end(); ++it){
            cout << "(" << it->first << " " << it->second << ")" << endl;
        }

    }

    return hypotheses;
}

Eigen::VectorXd cutGraph(Eigen::MatrixXd normalizedLaplacianMatrix, std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *voxels, std::multimap<int, int> *adjacency){

    using namespace Eigen;

    // Get Eigenvectors and Eigenvalues
    MatrixXd eigenvectors_matrix;
    MatrixXd eigenvalues_matrix;
    eigenDecomposition(normalizedLaplacianMatrix, eigenvectors_matrix, eigenvalues_matrix);

    // Initialize labels with connected components
    std::map<int, std::vector<int> > adjacency_map = convertMultimapToMap<int,int>(adjacency);
    std::map<int, std::vector<int> > connected_components = findConnectedComponents(&adjacency_map, voxels);
    printMap(&connected_components, "Connected Components");


    // Loop over the eigenvectors to discover cut hypotheses
    MatrixXd sorted_vectors = sortEigenvectors(eigenvectors_matrix, eigenvalues_matrix);
    std::multimap<int,int> cut_hypotheses = getGraphCutHypotheses(sorted_vectors, adjacency);


    // Evaluate Hypotheses to select the best cut candidates



    // Return labels of each vertex


}



Eigen::VectorXd kmeans_clustering(Eigen::MatrixXd normalizedLaplacianMatrix){

    using namespace Eigen;


    // Get Eigenvectors and Eigenvalues
    MatrixXd eigenvectors_matrix;
    MatrixXd eigenvalues_matrix;
    eigenDecomposition(normalizedLaplacianMatrix, eigenvectors_matrix, eigenvalues_matrix);
    int OriginalNumberOfClusters = eigenvectors_matrix.rows();


    MatrixXd test;
    test.resize(3,3);
    test.setZero();
    test(0,0) = 1 / sqrt(4);
    test(1,1) = 1 / sqrt(6);
    test(2,2) = 1 / sqrt(9);
    EigenSolver<MatrixXd> eig(test);
    cout << "TEST EIGENVALUES : "  << endl;
    cout << eig.eigenvalues() << endl;
    cout << "TEST EIGENVECTORS : " << endl;
    cout << eig.eigenvectors() << endl;


    // Calculate number of eigenvectors to keep
    int NumberOfEigenvectors = ceil(OriginalNumberOfClusters / 2);
    if(NumberOfEigenvectors > 12) NumberOfEigenvectors = 12;

    cv::Mat samples = retrieveKsmallestEigenvectors(eigenvectors_matrix, eigenvalues_matrix, NumberOfEigenvectors);

    cv::Mat labels;
    int attempts = 5;
    cv::Mat centers;


    std::vector<double> kmeans_score;
    cout << "kmeans_scores = [ " << endl;

    bool roundToZero = false;
    for(int k = 2; k < OriginalNumberOfClusters; k++){
        if(!roundToZero){
            double distance = cv::kmeans(samples, k, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.00001), attempts, cv::KMEANS_PP_CENTERS, centers );
            kmeans_score.push_back(distance);
            if(distance < 0.001) roundToZero = true;
            //cout << "Kmeans score (k = " << k << ") : " << distance << endl;
            cout << k << " " << distance << ";" << endl;
        }
        else{
            double distance = 0.0;
            kmeans_score.push_back(distance);
            cout << k << " " << distance << ";" << endl;
        }
    }
    cout << "];" << endl;


    int numberOfClusters = findElbowIndex(&kmeans_score) + 2;
    //numberOfClusters  = 4;
    cout << "From compactness, " << numberOfClusters << " clusters were found in the graph. " << endl;
    cv::kmeans(samples, numberOfClusters, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.00001), attempts, cv::KMEANS_PP_CENTERS, centers );


    cout << "Output Kmeans(" << KMEANS_NUMBER_OF_CLUSTERS << ") :" << endl;
    cout << labels << endl;

    VectorXd output_labels;
    output_labels.resize(OriginalNumberOfClusters);
    for(int i=0; i < OriginalNumberOfClusters; i++){
        output_labels(i) = labels.at<int>(i,0);
    }

    //cout << output_labels << endl;


    return output_labels;

}



void printSuperVoxels(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> *voxels, std::multimap<int, int> *adjacency, boost::shared_ptr<pcl::visualization::PCLVisualizer> visual, bool showGraph){

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

    // Print Facets
    int i = 0;
    int size = 0;
    for(std::map<int, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr >::iterator it = voxels->begin(); it != voxels->end(); ++it){
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc = it->second->voxels_;
        cout << "PC Size (" << it->first << ") : " << pc->size() << endl;
        size += pc->size();
        pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZRGBA> randColor(pc);
        std::stringstream ss;
        ss << i;
        std::string ind = ss.str();
        std::string pc_name = "object_" + ind;
        visual->addPointCloud<pcl::PointXYZRGBA>(pc,randColor,pc_name);
        visual->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, pc_name);
        i++;
    }
    cout << "Total Number of points : " << size << endl;


    // Print Graph Connectivity
    if(showGraph){
        std::multimap<int,int>::iterator label_itr = adjacency->begin ();
        for ( ; label_itr != adjacency->end (); )
        {
            //First get the label
            int supervoxel_label = label_itr->first;
            //Now get the supervoxel corresponding to the label
            pcl::Supervoxel<PointT>::Ptr supervoxel = voxels->at(supervoxel_label);

            //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
            PointCloudT adjacent_supervoxel_centers;
            std::multimap<int,int>::iterator adjacent_itr = adjacency->equal_range (supervoxel_label).first;
            for ( ; adjacent_itr!=adjacency->equal_range (supervoxel_label).second; ++adjacent_itr)
            {
                pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = voxels->at (adjacent_itr->second);
                adjacent_supervoxel_centers.push_back (neighbor_supervoxel->centroid_);
            }
            //Now we make a name for this polygon
            std::stringstream ss;
            ss << "supervoxel_" << supervoxel_label;
            // basically generates a "star" polygon mesh from the points given
            addSupervoxelConnectionsToViewer (supervoxel->centroid_, adjacent_supervoxel_centers, ss.str (), visual);
            //Move iterator forward to next label
            label_itr = adjacency->upper_bound (supervoxel_label);
        }
    }

}

void printMap(std::map<int, std::vector<int> > *myMap, std::string mapName){

    cout << "------" << mapName << "--------" << endl;
    for(std::map<int, std::vector<int> >::iterator it = myMap->begin(); it != myMap->end(); ++it){
        int key = it->first;

        cout << key << " : " ;
        std::vector<int> v = it->second;
        for(int i=0; i < v.size(); i++){
            cout << v.at(i) << ", ";
        }
        cout << endl;
    }

    cout << "---------------------" << endl;
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

void printEigenMatrix(Eigen::MatrixXd matrix, std::string name){
    cout << name << endl;
    cout << "[ " << endl;
    for(int i=0; i < matrix.rows(); i++){
        for(int j=0; j < matrix.cols(); j++){
            cout << matrix(i,j) << " ";
        }
        cout << ";" << endl;
    }
    cout << "]" << endl;
}

int main (int argc, char** argv){


    pcl::PointCloud<PointT>::Ptr scene_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr model_cloud(new pcl::PointCloud<PointT>);

    pcl::io::loadPCDFile("../bmw_clutter_remaining.pcd", *scene_cloud);
   // pcl::io::loadPCDFile("/home/jp/Downloads/OSD-0.2/pcd/test58.pcd", *scene_cloud);

    //    pcl::io::loadPCDFile("../customAlignment_fine.pcd", *model_cloud);

    initPCLViewer();



    /// COMMENT IF NOT USING OSD DATASET
//    pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
//    pcl::PassThrough<PointT> pass_filter;
//    pass_filter.setFilterFieldName("z");
//    pass_filter.setFilterLimits(0, 1.3);
//    pass_filter.setInputCloud(scene_cloud);
//    pass_filter.filter(*temp_cloud);
//    scene_cloud = extractPlane(temp_cloud, true);




    pclViewer->addPointCloud (scene_cloud, ColorHandlerT(scene_cloud, 255.0, 255.0, 0.0), "scene");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");



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
    superVoxels_clustering(scene_cloud_xyzrgb);


//    cout << "Pointcloud Coordinates : " << endl;
//    cout << "Pts = [ ";
//    for(int i = 0; i < scene_cloud_xyzrgb->size(); i++){
//        cout << scene_cloud_xyzrgb->at(i).x << " ";
//        cout << scene_cloud_xyzrgb->at(i).y << " ";
//        cout << scene_cloud_xyzrgb->at(i).z << ";" << endl;
//    }
//    cout << "]" << endl;


    // Find Primitives
    //    findCylinderPrimitive(scene_segmented);



    while (!pclViewer3->wasStopped()) {
        pclViewer->spinOnce (100);
        pclViewer2->spinOnce (100);
        pclViewer3->spinOnce (100);
    }

    return 0;
}
