#include <limits>
#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>

typedef pcl::PointXYZ PointT;

pcl::PointCloud<PointT>::Ptr computeUniformSampling(pcl::PointCloud<PointT>::Ptr p_cloudIn, double radius);
double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud);
pcl::PointCloud<PointT>::Ptr computeISSKeypoints(pcl::PointCloud<PointT>::Ptr cloud);

double scene_sampling_radius = 0.01;
double model_sampling_radius = 0.01;

class FeatureCloud
{
public:
    // A bit of shorthand
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
    typedef pcl::PointCloud<pcl::Normal> SurfaceNormals;
    typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
    typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

    FeatureCloud () :
        search_method_xyz_ (new SearchMethod),
        normal_radius_ (0.02f),
        feature_radius_ (0.02f)
    {}

    ~FeatureCloud () {}

    // Process the given cloud
    void
    setInputCloud (PointCloud::Ptr xyz)
    {
        xyz_ = xyz;
        processInput ();
    }

    // Load and process the cloud in the given PCD file
    void
    loadInputCloud (const std::string &pcd_file)
    {
        std::cout << "Loading file : " << pcd_file << std::endl;
        xyz_ = PointCloud::Ptr (new PointCloud);
        pcl::io::loadPCDFile (pcd_file, *xyz_);
        xyz_ = computeUniformSampling(xyz_, model_sampling_radius);
        //xyz_ = computeISSKeypoints(xyz_);

        processInput ();
    }

    // Get a pointer to the cloud 3D points
    PointCloud::Ptr
    getPointCloud () const
    {
        return (xyz_);
    }

    // Get a pointer to the cloud of 3D surface normals
    SurfaceNormals::Ptr
    getSurfaceNormals () const
    {
        return (normals_);
    }

    // Get a pointer to the cloud of feature descriptors
    LocalFeatures::Ptr
    getLocalFeatures () const
    {
        return (features_);
    }

protected:
    // Compute the surface normals and local features
    void
    processInput ()
    {
        computeSurfaceNormals ();
        computeLocalFeatures ();
    }

    // Compute the surface normals
    void
    computeSurfaceNormals ()
    {
        normals_ = SurfaceNormals::Ptr (new SurfaceNormals);

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
        norm_est.setInputCloud (xyz_);
        norm_est.setSearchMethod (search_method_xyz_);
        norm_est.setRadiusSearch (normal_radius_);
        norm_est.compute (*normals_);
    }

    // Compute the local feature descriptors
    void
    computeLocalFeatures ()
    {
        features_ = LocalFeatures::Ptr (new LocalFeatures);

        pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
        fpfh_est.setInputCloud (xyz_);
        fpfh_est.setInputNormals (normals_);
        fpfh_est.setSearchMethod (search_method_xyz_);
        fpfh_est.setRadiusSearch (feature_radius_);
        fpfh_est.compute (*features_);
    }

private:
    // Point cloud data
    PointCloud::Ptr xyz_;
    SurfaceNormals::Ptr normals_;
    LocalFeatures::Ptr features_;
    SearchMethod::Ptr search_method_xyz_;

    // Parameters
    float normal_radius_;
    float feature_radius_;
};

class TemplateAlignment
{
public:

    // A struct for storing alignment results
    struct Result
    {
        float fitness_score;
        Eigen::Matrix4f final_transformation;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    TemplateAlignment () :
        min_sample_distance_ (0.05f),
        max_correspondence_distance_ (0.08f*0.08f),
        nr_iterations_ (500)
    {
        // Intialize the parameters in the Sample Consensus Intial Alignment (SAC-IA) algorithm
        sac_ia_.setMinSampleDistance (min_sample_distance_);
        sac_ia_.setMaxCorrespondenceDistance (max_correspondence_distance_);
        sac_ia_.setMaximumIterations (nr_iterations_);
    }

    ~TemplateAlignment () {}

    // Set the given cloud as the target to which the templates will be aligned
    void
    setTargetCloud (FeatureCloud &target_cloud)
    {
        target_ = target_cloud;
        sac_ia_.setInputTarget (target_cloud.getPointCloud ());
        sac_ia_.setTargetFeatures (target_cloud.getLocalFeatures ());
    }

    // Add the given cloud to the list of template clouds
    void
    addTemplateCloud (FeatureCloud &template_cloud)
    {
        templates_.push_back (template_cloud);
    }

    // Align the given template cloud to the target specified by setTargetCloud ()
    void
    align (FeatureCloud &template_cloud, TemplateAlignment::Result &result)
    {
        sac_ia_.setInputSource( template_cloud.getPointCloud ());
        sac_ia_.setSourceFeatures (template_cloud.getLocalFeatures ());

        pcl::PointCloud<pcl::PointXYZ> registration_output;
        sac_ia_.align (registration_output);

        result.fitness_score = (float) sac_ia_.getFitnessScore (max_correspondence_distance_);
        result.final_transformation = sac_ia_.getFinalTransformation ();
    }

    // Align all of template clouds set by addTemplateCloud to the target specified by setTargetCloud ()
    void
    alignAll (std::vector<TemplateAlignment::Result, Eigen::aligned_allocator<Result> > &results)
    {
        results.resize (templates_.size ());
        for (size_t i = 0; i < templates_.size (); ++i)
        {
            align (templates_[i], results[i]);
        }
    }

    // Align all of template clouds to the target cloud to find the one with best alignment score
    int
    findBestAlignment (TemplateAlignment::Result &result)
    {
        // Align all of the templates to the target cloud
        std::vector<Result, Eigen::aligned_allocator<Result> > results;
        alignAll (results);

        // Find the template with the best (lowest) fitness score
        float lowest_score = std::numeric_limits<float>::infinity ();
        int best_template = 0;
        for (size_t i = 0; i < results.size (); ++i)
        {
            const Result &r = results[i];
            if (r.fitness_score < lowest_score)
            {
                lowest_score = r.fitness_score;
                best_template = (int) i;
            }
        }

        // Output the best alignment
        result = results[best_template];
        return (best_template);
    }

private:
    // A list of template clouds and the target to which they will be aligned
    std::vector<FeatureCloud> templates_;
    FeatureCloud target_;

    // The Sample Consensus Initial Alignment (SAC-IA) registration routine and its parameters
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
    float min_sample_distance_;
    float max_correspondence_distance_;
    int nr_iterations_;
};



// This function by Tommaso Cavallari and Federico Tombari, taken from the tutorial
// http://pointclouds.org/documentation/tutorials/correspondence_grouping.php
double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
    double resolution = 0.0;
    int numberOfPoints = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> squaredDistances(2);
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (! pcl_isfinite((*cloud)[i].x))
            continue;

        // Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
        if (nres == 2)
        {
            resolution += sqrt(squaredDistances[1]);
            ++numberOfPoints;
        }
    }
    if (numberOfPoints != 0)
        resolution /= numberOfPoints;

    return resolution;
}

pcl::PointCloud<PointT>::Ptr computeISSKeypoints(pcl::PointCloud<PointT>::Ptr cloud)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
    detector.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    detector.setSearchMethod(kdtree);
    double resolution = computeCloudResolution(cloud);
    //double resolution = 0.005;

    // Set the radius of the spherical neighborhood used to compute the scatter matrix.
    detector.setSalientRadius(6 * resolution);
    // Set the radius for the application of the non maxima supression algorithm.
    detector.setNonMaxRadius(4 * resolution);
    // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
    detector.setMinNeighbors(5);
    // Set the upper bound on the ratio between the second and the first eigenvalue.
    detector.setThreshold21(0.975);
    // Set the upper bound on the ratio between the third and the second eigenvalue.
    detector.setThreshold32(0.975);
    // Set the number of prpcessing threads to use. 0 sets it to automatic.
    detector.setNumberOfThreads(4);

    detector.setNormalRadius (4 * resolution);
    detector.setBorderRadius (1 * resolution);

    detector.compute(*keypoints);


    return keypoints;

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


    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_cloud (new pcl::PointCloud<pcl::PointXYZ>);

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

pcl::PointCloud<PointT>::Ptr extractPlane(pcl::PointCloud<PointT>::Ptr cloud, bool inliers){

    pcl::PointCloud<pcl::PointXYZ>::Ptr returned_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inlierIndices(new pcl::PointIndices);

    // Find the plane coefficients from the model
    pcl::SACSegmentation<pcl::PointXYZ> segmentation;
    segmentation.setInputCloud(cloud);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(0.01);
    segmentation.setOptimizeCoefficients(true);
    segmentation.segment(*inlierIndices, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inlierIndices);
    extract.setNegative (!inliers);
    extract.filter (*returned_cloud);

    return returned_cloud;

}


// Align a collection of object templates to a sample point cloud
int
main (int argc, char **argv)
{
    if (argc < 3)
    {
        printf ("No target PCD file given!\n");
        return (-1);
    }

    time_t timer_beginning;
    time_t timer_end;
    time(&timer_beginning);

    // Load the object templates specified in the object_templates.txt file
    std::vector<FeatureCloud> object_templates;
    std::ifstream input_stream (argv[1]);
    object_templates.resize (0);
    std::string pcd_filename;
    while (input_stream.good ())
    {
        std::getline (input_stream, pcd_filename);
        if (pcd_filename.empty () || pcd_filename.at (0) == '#') // Skip blank lines or comments
            continue;

        FeatureCloud template_cloud;
        template_cloud.loadInputCloud (pcd_filename);
        object_templates.push_back (template_cloud);
    }
    input_stream.close ();

    // Load the target cloud PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile (argv[2], *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_unfiltered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *scene_unfiltered);

    // Preprocess the cloud by...
    // ...removing distant points
    const float depth_limit = 2.5;
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, depth_limit);
    pass.filter (*cloud);

    // ... and downsampling the point cloud
    const float voxel_grid_size = 0.005f;
    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setInputCloud (cloud);
    vox_grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    //vox_grid.filter (*cloud); // Please see this http://www.pcl-developers.org/Possible-problem-in-new-VoxelGrid-implementation-from-PCL-1-5-0-td5490361.html
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);
    vox_grid.filter (*tempCloud);
    *cloud = *tempCloud;

    //Uniform Sampling
    cloud = computeUniformSampling(cloud, scene_sampling_radius);
    //cloud = computeISSKeypoints(cloud);




    // Assign to the target FeatureCloud
    FeatureCloud target_cloud;
    target_cloud.setInputCloud (cloud);

    // Set the TemplateAlignment inputs
    TemplateAlignment template_align;
    for (size_t i = 0; i < object_templates.size (); ++i)
    {
        template_align.addTemplateCloud (object_templates[i]);
    }
    template_align.setTargetCloud (target_cloud);

    // Find the best template alignment
    TemplateAlignment::Result best_alignment;
    int best_index = template_align.findBestAlignment (best_alignment);
    const FeatureCloud &best_template = object_templates[best_index];

    // Print the alignment fitness score (values less than 0.00002 are good)
    printf ("Best fitness score: %f\n", best_alignment.fitness_score);

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = best_alignment.final_transformation.block<3,3>(0, 0);
    Eigen::Vector3f translation = best_alignment.final_transformation.block<3,1>(0, 3);

    printf ("\n");
    printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));



    // ITERATIVE CLOSEST POINT (ICP)
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(object_templates[0].getPointCloud());
    icp.setInputTarget(cloud);
    icp.setMaxCorrespondenceDistance(0.1);
    icp.setMaximumIterations(40);
    pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>());
    icp.align(*Final,best_alignment.final_transformation);
    Eigen::Matrix4f icp_transform = icp.getFinalTransformation();

    // Save the aligned template for visualization
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_icp (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::transformPointCloud (*best_template.getPointCloud (), *transformed_cloud, best_alignment.final_transformation);
    pcl::transformPointCloud (*object_templates[0].getPointCloud(), *transformed_cloud_icp, icp_transform);

    //pcl::io::savePCDFileBinary ("output.pcd", *transformed_cloud);
    //pcl::io::savePCDFileBinary ("output2.pcd", *transformed_cloud_icp);

    time(&timer_end);
    double seconds = difftime(timer_end,timer_beginning);
    std::cout << "It took " << seconds << " seconds to complete the alignment!" << std::endl;




//    // FIND PLANE IN ALIGNED MODEL
//    pcl::PointCloud<pcl::PointXYZ>::Ptr model_plane(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_outliers(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//    pcl::PointIndices::Ptr inlierIndices(new pcl::PointIndices);

//    // Find the plane coefficients from the model
//    pcl::SACSegmentation<pcl::PointXYZ> segmentation;
//    segmentation.setInputCloud(transformed_cloud_icp);
//    segmentation.setModelType(pcl::SACMODEL_PLANE);
//    segmentation.setMethodType(pcl::SAC_RANSAC);
//    segmentation.setDistanceThreshold(0.01);
//    segmentation.setOptimizeCoefficients(true);
//    segmentation.segment(*inlierIndices, *coefficients);

//    pcl::ExtractIndices<pcl::PointXYZ> extract;
//    extract.setInputCloud(transformed_cloud_icp);
//    extract.setIndices(inlierIndices);
//    extract.filter(*model_plane);

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_plane(new pcl::PointCloud<pcl::PointXYZ>);
    model_plane = extractPlane(transformed_cloud_icp, true);

    // FIND POLYGON OF THE PLANE (CONVEX OR CONVCAVE)
    pcl::PointCloud<pcl::PointXYZ>::Ptr concaveHull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConcaveHull<pcl::PointXYZ> hull;
    std::vector<pcl::Vertices> polygons;
    hull.setInputCloud(model_plane);
    hull.setAlpha(0.1);
    hull.reconstruct(*concaveHull, polygons);


    pcl::PointCloud<pcl::PointXYZ>::Ptr objects (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::CropHull<pcl::PointXYZ> bb_filter;

    bb_filter.setDim(2);
    bb_filter.setInputCloud(scene_unfiltered);
    bb_filter.setHullIndices(polygons);
    bb_filter.setHullCloud(concaveHull);
    bb_filter.filter(*objects);

    pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_objects (new pcl::PointCloud<pcl::PointXYZ>);
    segmented_objects = extractPlane(objects, false);


//    // Extract points in the the polygon
//    pcl::ExtractPolygonalPrismData<pcl::PointXYZ> ex;
//    ex.setInputCloud (scene_unfiltered);
//    ex.setInputPlanarHull (concaveHull);
//    pcl::PointIndices::Ptr output (new pcl::PointIndices);
//    ex.segment (*output);

//    pcl::ExtractIndices<pcl::PointXYZ> extract2;
//    pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
//    extract2.setInputCloud(scene_unfiltered);
//    extract2.setIndices(output);
//    extract2.filter(*object);

//    // From the remaining points, remove those who belong to the plane
//    // Use the plane coefficients from the model to find the plane in the original point cloud
//    pcl::SACSegmentation<pcl::PointXYZ> segmentation2;
//    pcl::PointIndices::Ptr inlierIndicesScene(new pcl::PointIndices);
//    segmentation2.setInputCloud(object);
//    segmentation2.setModelType(pcl::SACMODEL_PLANE);
//    segmentation2.setMethodType(pcl::SAC_RANSAC);
//    segmentation2.setDistanceThreshold(0.01);
//    segmentation2.setOptimizeCoefficients(false);
//    segmentation2.segment(*inlierIndicesScene, *coefficients);


//    pcl::PointCloud<pcl::PointXYZ>::Ptr object_extracted_planes(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::ExtractIndices<pcl::PointXYZ> extract3;
//    extract3.setInputCloud(object);
//    extract3.setIndices(inlierIndicesScene);
//    extract3.setNegative (true);
//    extract3.filter(*object_extracted_planes);



//    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//    tree->setInputCloud (cloud_filtered);

//    std::vector<pcl::PointIndices> cluster_indices;
//    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//    ec.setClusterTolerance (0.02); // 2cm
//    ec.setMinClusterSize (300);
//    ec.setMaxClusterSize (25000);
//    ec.setSearchMethod (tree);
//    ec.setInputCloud (cloud_filtered);
//    ec.extract (cluster_indices);



    //==============================TESTING 3D HULL=====================================//
//    pcl::ConvexHull<pcl::PointXYZ> hull;
//    pcl::PointCloud<pcl::PointXYZ>::Ptr surface_hull (new pcl::PointCloud<pcl::PointXYZ>);
//    hull.setInputCloud(transformed_cloud_icp);
//    hull.setDimension(2);
//    std::vector<pcl::Vertices> polygons;
//    hull.reconstruct(*surface_hull, polygons);


//    pcl::PointCloud<pcl::PointXYZ>::Ptr objects (new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::CropHull<pcl::PointXYZ> bb_filter;

//    bb_filter.setDim(2);
//    bb_filter.setInputCloud(scene_unfiltered);
//    bb_filter.setHullIndices(polygons);
//    bb_filter.setHullCloud(surface_hull);
//    bb_filter.filter(*objects);

//    // FIND PLANE IN ALIGNED MODEL
//    pcl::PointCloud<pcl::PointXYZ>::Ptr model_plane(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_outliers(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//    pcl::PointIndices::Ptr inlierIndices(new pcl::PointIndices);

//    // Find the plane coefficients from the model
//    pcl::SACSegmentation<pcl::PointXYZ> segmentation;
//    segmentation.setInputCloud(objects);
//    segmentation.setModelType(pcl::SACMODEL_PLANE);
//    segmentation.setMethodType(pcl::SAC_RANSAC);
//    segmentation.setDistanceThreshold(0.01);
//    segmentation.setOptimizeCoefficients(true);
//    segmentation.segment(*inlierIndices, *coefficients);


//    pcl::ExtractIndices<pcl::PointXYZ> extract;
//    extract.setInputCloud(objects);
//    extract.setIndices(inlierIndices);
//    extract.setNegative (true);
//    extract.filter(*plane_outliers);

    //==============================TESTING 3D HULL END=====================================//





    pcl::io::savePCDFileASCII("segmented_container.pcd", *objects);




//    // Visualize the segmented object at the end
//    pcl::visualization::PCLVisualizer visu("Alignment");
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(transformed_cloud_icp, 255.0, 0.0, 0.0);
//    visu.addPointCloud (transformed_cloud_icp, red, "scene");

//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> yellow(concaveHull, 255.0, 255.0, 0.0);
//    visu.addPointCloud (concaveHull, yellow, "plane");
//    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "plane");

//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pale_blue(segmented_objects, 0.0, 255.0, 255.0);
//    visu.addPointCloud (segmented_objects, pale_blue, "object");




    // Visualize the alignment
    pcl::visualization::PCLVisualizer visu("Alignment");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud, 255.0, 0.0, 0.0);
    visu.addPointCloud (cloud, red, "aligned_model");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> yellow(transformed_cloud, 255.0, 255.0, 0.0);
    visu.addPointCloud (transformed_cloud, yellow, "scene");


    // VISUALIZATION


    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(transformed_cloud, 0.0, 255.0, 0.0);
    //visu.addPointCloud (transformed_cloud, green, "sac-ia");

    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(transformed_cloud_icp, 0.0, 0.0, 255.0);
    //    visu.addPointCloud (transformed_cloud_icp, blue, "icp");





    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pale_blue(concaveHull, 0.0, 255.0, 255.0);
    //    visu.addPointCloud (concaveHull, pale_blue, "hull");
    //    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "hull");

//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pale_blue(segmented_objects, 0.0, 255.0, 255.0);
//    visu.addPointCloud (segmented_objects, pale_blue, "object");
//    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object");


    visu.spin ();

    return (0);
}
