// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#include <simple_object_segmentation/simple_object_segmentation.hpp>
#include <string>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/point_cloud_conversion.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl/features/normal_3d.h>
#include <pcl/conversions.h>

SimpleObjectSegmentation::SimpleObjectSegmentation() :
    num_threads_(8), neigbor_size_(26), cc_thresh_(-0.01f) {
    std::string type;
    this->pnh_.param<std::string>("user_input", this->user_in_type_, "none");
    this->pnh_.param<float>("cc_thresh", this->cc_thresh_, -0.01);
    ROS_INFO("user_input", this->user_in_type_, "none");

    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->onInit();
}

void SimpleObjectSegmentation::onInit() {
    ROS_INFO("onInit");
    this->subscribe();

    // ROS_INFO("hello custom message");
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cloud", 1);
    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>("/indices", 1);
    this->pub_centroid_ = this->pnh_.advertise<
       geometry_msgs::PointStamped>("/object_centroid", 1);
     this->pub_img_ = this->pnh_.advertise<sensor_msgs::Image>(
          "/segmented_im", 1);
}

void SimpleObjectSegmentation::subscribe() {
    ROS_INFO("subscribe");
    if (this->user_in_type_ == "rect") {
       ROS_INFO("\033[34mRUN TYPE RECT\033[0m");
       this->sub_point_.subscribe(this->pnh_, "points", 1);
       this->sub_rect_.subscribe(this->pnh_, "rect", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       this->sync_->connectInput(this->sub_point_, this->sub_rect_);
       this->sync_->registerCallback(
          boost::bind(&SimpleObjectSegmentation::callbackRect, this, _1, _2));
    } else if (this->user_in_type_ == "point") {
       ROS_INFO("\033[34mRUN TYPE POINT\033[0m");
       this->sub_point_.subscribe(this->pnh_, "points", 1);
       this->sub_screen_pt_.subscribe(this->pnh_, "screen_pt", 1);
       this->sync2_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy2> >(100);
       this->sync2_->connectInput(this->sub_point_, this->sub_screen_pt_);
       this->sync2_->registerCallback(
          boost::bind(&SimpleObjectSegmentation::callbackPoint, this, _1, _2));
    } else {
       ROS_INFO("\033[34mRUN TYPE AUTO\033[0m");
       this->sub_cloud_ = this->pnh_.subscribe(
          "points", 1, &SimpleObjectSegmentation::callback, this);
    }
}

void SimpleObjectSegmentation::unsubscribe() {
    ROS_INFO("unsubscribe");
    if (this->user_in_type_ == "rect") {
       this->sub_point_.unsubscribe();
       this->sub_rect_.unsubscribe();
    } else {
       this->sub_cloud_.shutdown();
    }
}

void SimpleObjectSegmentation::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    ROS_INFO("CALL BACK");
    ROS_ERROR("SAMPLE ERROR");
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
       ROS_ERROR("[::cloudCB]: EMPTY INPUTS");
       return;
    }
    this->header_ = cloud_msg->header;

    assert((cloud->width != 1 || cloud->height != 1)  &&
           "\033[31m UNORGANIZED INPUT CLOUD \033[0m");

    std::cout<<"point cloud:"<<*cloud<<std::endl;

    pcl::PointXYZRGB minPt, maxPt;
    pcl::getMinMax3D (*cloud, minPt, maxPt);
    ROS_INFO("Let's print!");
    std::cout<<"Max x: "<<maxPt.x<<" Max y: "<<maxPt.y<<" Max z: "<<maxPt.z<<std::endl;
    std::cout<<"Min x: "<<minPt.x<<" Min y: "<<minPt.y<<" Min z: "<<minPt.z<<std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudFiltered (new pcl::PointCloud<pcl::PointXYZRGB>);

    // Define min and max for X, Y and Z
    float minX = -0.5, minY = -0.4, minZ = 0.5;
    float maxX = +0.1, maxY = 0.0 /*+0.4*/, maxZ = +1.0;

    pcl::CropBox<pcl::PointXYZRGB> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(cloud);
    boxFilter.filter(*cloudFiltered);
    // boxFilter.applyFilter(cloudFiltered);
    //cloudFiltered->width = 500;
    //cloudFiltered->height = 500;

    std::cout<<"point cloud filtered:"<<cloudFiltered->width<<" "<<cloudFiltered->height<<std::endl;

    this->segment(cloudFiltered);
    // this->segment(cloud);
    ROS_INFO("CALL BACK SUCCESSFUL");
}

void SimpleObjectSegmentation::callbackRect(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PolygonStamped::ConstPtr &rect_msg) {
    ROS_INFO("CALL BACK RECT");
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty() || rect_msg->polygon.points.size() == 0) {
       ROS_ERROR("[::cloudCB]: EMPTY INPUTS");
       return;
    }
    assert((cloud->width != 1 || cloud->height != 1)  &&
          "\033[31m UNORGANIZED INPUT CLOUD \033[0m");

    for (int i=1; i<20; i++) {
        this->segment(cloud);
    }

    this->input_size_ = cv::Size(cloud->width, cloud->height);

    int x = rect_msg->polygon.points[0].x;
    int y = rect_msg->polygon.points[0].y;
    int width = rect_msg->polygon.points[1].x - x;
    int height = rect_msg->polygon.points[1].y - y;

    std::cout<<"x, y, width, height "<<x<<" "<<y<<" "<<width<<" "<<height<<std::endl;

    x -= width/2;
    y -= height/2;
    width *= 2;
    height *= 2;

    x = x < 0 ? 0 : x;
    y = y < 0 ? 0 : y;
    width -= x + width > cloud->width ? (x + width) - cloud->width : 0;
    height -= y + height > cloud->height ? (y + height) - cloud->height : 0;
    this->rect_ = cv::Rect_<int>(x, y, width, height);
    std::cout<<"Rect x, y, width, height "<<x<<" "<<y<<" "<<width<<" "<<height<<std::endl;
    std::cout<<"Cloud width, height "<<cloud->width<<" "<<cloud->height<<std::endl;

    PointNormal::Ptr normals(new PointNormal);
    this->getNormals(normals, cloud);

    PointCloudNormal::Ptr src_pts(new PointCloudNormal);
    this->fastSeedRegionGrowing(src_pts, cloud, normals);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*src_pts, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    // this->pub_cloud_.publish(ros_cloud);
    ROS_INFO("CALL BACK RECT SUCCESSFUL");
}

void SimpleObjectSegmentation::callbackPoint(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PointStamped::ConstPtr &screen_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
       ROS_ERROR("[::cloudCB]: EMPTY INPUTS");
       return;
    }

    assert((cloud->width != 1 || cloud->height != 1)  &&
          "\033[31m UNORGANIZED INPUT CLOUD \033[0m");

    int seed_index = screen_msg->point.x + (cloud->width * screen_msg->point.y);
    this->seed_point_ = cloud->points[seed_index];
    if (std::isnan(seed_point_.x) || std::isnan(seed_point_.x) ||
        std::isnan(seed_point_.x)) {
       ROS_ERROR("SELETED POINT IS NAN");
       return;
    }

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, nan_indices);

    double dist = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < cloud->size(); i++) {
       double d = pcl::distances::l2(cloud->points[i].getVector4fMap(),
                                     seed_point_.getVector4fMap());
       if (d < dist) {
          dist = d;
          idx = i;
       }
    }

    PointNormal::Ptr normals(new PointNormal);
    this->getNormals<float>(cloud, normals, 0.03f, false);

    seed_index = idx;
    this->seed_point_ = cloud->points[seed_index];
    this->seed_normal_ = normals->points[seed_index];

    if (cloud->size() != normals->size()) {
       ROS_ERROR("CLOUD AND NORMALS SIZES ARE DIFF");
       return;
    }

    std::vector<int> labels(static_cast<int>(cloud->size()), -1);

    if (labels.size() < seed_index) {
       ROS_WARN("INCORRECT SIZES");
       return;
    }
    labels[seed_index] = 1;

    this->kdtree_->setInputCloud(cloud);
    this->seedCorrespondingRegion(labels, cloud, normals, seed_index);

    PointCloud::Ptr in_cloud(new PointCloud);
    std::vector<pcl::PointIndices> all_indices(1);
    int icounter = 0;

    float min_z = FLT_MAX;
    float min_y = FLT_MAX;
    float min_x = FLT_MAX;

    float max_z = 0.0f;
    float max_y = 0.0f;
    float max_x = 0.0f;

    float center_x = 0.0f;
    float center_y = 0.0f;
    float center_z = 0.0f;

    for (int i = 0; i < labels.size(); i++) {
       if (labels[i] != -1) {
          in_cloud->push_back(cloud->points[i]);
          all_indices[0].indices.push_back(icounter++);

          min_z = cloud->points[i].z < min_z ? cloud->points[i].z : min_z;
          min_y = cloud->points[i].y < min_y ? cloud->points[i].y : min_y;
          min_x = cloud->points[i].x < min_x ? cloud->points[i].x : min_x;

          max_z = cloud->points[i].z > max_z ? cloud->points[i].z : max_z;
          max_y = cloud->points[i].y > max_y ? cloud->points[i].y : max_y;
          max_x = cloud->points[i].x > max_x ? cloud->points[i].x : max_x;

          center_z += cloud->points[i].z;
          center_y += cloud->points[i].y;
          center_x += cloud->points[i].x;
       }
    }
    center_x /= static_cast<float>(in_cloud->size());
    center_y /= static_cast<float>(in_cloud->size());
    center_z /= static_cast<float>(in_cloud->size());

    center_z = min_z + (max_z - min_z) / 2.0f;
    center_y = min_y + (max_y - min_y) / 2.0f;
    // center_x = min_x - (max_x - min_x) / 2.0f;


    //! cluster points indices
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = this->convertToROSPointIndices(
       all_indices, cloud_msg->header);
    ros_indices.header = cloud_msg->header;

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*in_cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    this->pub_cloud_.publish(ros_cloud);
    this->pub_indices_.publish(ros_indices);

    //! publish object centeroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*in_cloud, centroid);
    geometry_msgs::PointStamped ros_point;
    ros_point.header = cloud_msg->header;
    // ros_point.point.x = centroid(0);
    // ros_point.point.y = centroid(1);
    // ros_point.point.z = centroid(2);

    ros_point.point.x = center_x;
    ros_point.point.y = center_y;
    ros_point.point.z = center_z;
    this->pub_centroid_.publish(ros_point);
}

void SimpleObjectSegmentation::seedCorrespondingRegion(
    std::vector<int> &labels, const PointCloud::Ptr cloud,
    const PointNormal::Ptr normals, const int parent_index) {
    std::vector<int> neigbor_indices;
    this->getPointNeigbour<int>(neigbor_indices, cloud,
                                cloud->points[parent_index],
                                this->neigbor_size_);

    int neigb_lenght = static_cast<int>(neigbor_indices.size());
    std::vector<int> merge_list(neigb_lenght);
    merge_list[0] = -1;

#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)        \
    shared(merge_list, labels)
#endif
    for (int i = 1; i < neigbor_indices.size(); i++) {
       int index = neigbor_indices[i];
       if (index != parent_index && labels[index] == -1) {
          Eigen::Vector4f parent_pt = cloud->points[
             parent_index].getVector4fMap();
          Eigen::Vector4f parent_norm = normals->points[
             parent_index].getNormalVector4fMap();
          Eigen::Vector4f child_pt = cloud->points[index].getVector4fMap();
          Eigen::Vector4f child_norm = normals->points[
             index].getNormalVector4fMap();
          if (this->seedVoxelConvexityCriteria(
                 parent_norm, child_pt, child_norm,
                 this->cc_thresh_) == 1) {
             merge_list[i] = index;
             labels[index] = 1;
          } else {
             merge_list[i] = -1;
          }
       } else {
          merge_list[i] = -1;
       }
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) schedule(guided, 1)
#endif
    for (int i = 0; i < merge_list.size(); i++) {
       int index = merge_list[i];
       if (index != -1) {
          seedCorrespondingRegion(labels, cloud, normals, index);
       }
    }
}

template<class T>
void SimpleObjectSegmentation::getPointNeigbour(
    std::vector<int> &neigbor_indices, const PointCloud::Ptr cloud,
    const PointT seed_point_, const T K, bool is_knn) {
    if (cloud->empty() || std::isnan(seed_point_.x) ||
        std::isnan(seed_point_.y) || std::isnan(seed_point_.z)) {
       ROS_ERROR("THE CLOUD IS EMPTY. RETURING VOID IN GET NEIGBOUR");
       return;
    }
    neigbor_indices.clear();
    std::vector<float> point_squared_distance;
    if (is_knn) {
       int search_out = kdtree_->nearestKSearch(
          seed_point_, K, neigbor_indices, point_squared_distance);
    } else {
       int search_out = kdtree_->radiusSearch(
          seed_point_, K, neigbor_indices, point_squared_distance);
    }
}

void SimpleObjectSegmentation::getNormals(
    PointNormal::Ptr normals, const PointCloud::Ptr cloud) {
    if (cloud->empty()) {
       ROS_ERROR("-Input cloud is empty in normal estimation");
       return;
    }
    this->ne_.setNormalEstimationMethod(this->ne_.AVERAGE_3D_GRADIENT);
    this->ne_.setMaxDepthChangeFactor(0.02f);
    this->ne_.setNormalSmoothingSize(10.0f);
    this->ne_.setInputCloud(cloud);
    this->ne_.compute(*normals);
}

template<class T>
void SimpleObjectSegmentation::getNormals(
    const PointCloud::Ptr cloud, PointNormal::Ptr normals,
    const T k, bool use_knn) const {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: The Input cloud is Empty.....");
       return;
    }
    pcl::NormalEstimationOMP<PointT, NormalT> ne;
    ne.setInputCloud(cloud);
    ne.setNumberOfThreads(this->num_threads_);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    ne.setSearchMethod(tree);
    if (use_knn) {
       ne.setKSearch(k);
    } else {
       ne.setRadiusSearch(k);
    }
    ne.compute(*normals);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr SimpleObjectSegmentation::passThroughFilter1D(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const std::string field, const double low, const double high) {
    if (low > high) {
        std::cout << "Warning! Min is greater than max!\n";
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;

    pass.setInputCloud(cloud);
    pass.setFilterFieldName(field);
    pass.setFilterLimits(low, high);
    pass.setFilterLimitsNegative(false); // don't remove inside
    pass.filter(*cloud_filtered);
    return cloud_filtered;
}

cv::Mat SimpleObjectSegmentation::makeImageFromPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float stepSize1, float stepSize2) {

    pcl::PointXYZRGB cloudMin, cloudMax;
    pcl::getMinMax3D(*cloud, cloudMin, cloudMax);

    std::string dimen1, dimen2;
    float dimen1Max, dimen1Min, dimen2Min, dimen2Max;
    dimen1 = "x";
    dimen2 = "y";
    dimen1Min = cloudMin.x;
    dimen1Max = cloudMax.x;
    dimen2Min = cloudMin.y;
    dimen2Max = cloudMax.y;

    std::vector<std::vector<int>> pointCountGrid;
    int maxPoints = 0;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> grid;

    for (float i = dimen1Min; i < dimen1Max; i += stepSize1) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr slice = passThroughFilter1D(cloud, dimen1, i, i + stepSize1);
        grid.push_back(slice);
        std::vector<int> slicePointCount;
        for (float j = dimen2Min; j < dimen2Max; j += stepSize2) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr grid_cell = passThroughFilter1D(slice, dimen2, j, j + stepSize2);
            // std::cout<<"grid_cell: "<<grid_cell->x<<" "<<grid_cell->y<<" "<<grid_cell->z<<std::endl;
            int gridSize = grid_cell->size();
            slicePointCount.push_back(gridSize);
            if (gridSize > maxPoints) {
                maxPoints = gridSize;
            }
        }
        pointCountGrid.push_back(slicePointCount);
    }
    cv::Mat mat(static_cast<int>(pointCountGrid.size()), static_cast<int>(pointCountGrid.at(0).size()), CV_8UC1);
    mat = cv::Scalar(0);

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            int pointCount = pointCountGrid.at(i).at(j);
            float percentOfMax = (pointCount + 0.0) / (maxPoints + 0.0);
            int intensity = percentOfMax * 255;
            mat.at<uchar>(i, j) = intensity;
        }
    }
    return mat;
}

void SimpleObjectSegmentation::segment(const PointCloud::Ptr in_cloud) {
    SupervoxelMap supervoxel_clusters;
    AdjacentList adjacency_list;
    this->supervoxelSegmentation(in_cloud, supervoxel_clusters, adjacency_list);
    UInt32Map voxel_labels;
    for (auto it = supervoxel_clusters.begin(); it != supervoxel_clusters.end(); it++) {
       voxel_labels[it->first] = -1;
    }

    int label = 0;
    for (auto itr = adjacency_list.begin(); itr != adjacency_list.end();) {
       int32_t vindex = itr->first;
       if (voxel_labels[vindex] == -1) {
          if (!supervoxel_clusters.at(vindex)->voxels_->empty()) {
             voxel_labels[vindex] = label;
             this->segmentRecursiveCC(voxel_labels, label, adjacency_list,
                  supervoxel_clusters, vindex);
             label++;
          }
       }
       itr = adjacency_list.upper_bound(vindex);
    }

    SupervoxelMap sv_clustered;
    AdjacentList update_adjlist;

    //! initalization
    for (int i = 0; i < label + 1; i++) {
       pcl::Supervoxel<PointT>::Ptr tmp_sv(new pcl::Supervoxel<PointT>);
       sv_clustered[i] = tmp_sv;
    }

    pcl::Supervoxel<PointT>::Ptr tmp_sv(new pcl::Supervoxel<PointT>);
    for (auto it = voxel_labels.begin(); it != voxel_labels.end(); it++) {
       if (it->second != -1) {
          *(sv_clustered[it->second]->voxels_) +=
             *supervoxel_clusters.at(it->first)->voxels_;
          *(sv_clustered[it->second]->normals_) +=
             *supervoxel_clusters.at(it->first)->normals_;

          auto v_label = it->second;
          for (auto it2 = adjacency_list.equal_range(it->first).first;
               it2 != adjacency_list.equal_range(it->first).second; ++it2) {
             auto n_label = voxel_labels[it2->second];
             if (n_label != v_label) {
                update_adjlist.insert(std::make_pair(it->second, n_label));
             }
          }
       }
    }

    sensor_msgs::PointCloud2 ros_voxels;
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pre_im = this->publishSupervoxel(sv_clustered,
                           ros_voxels, ros_indices,
                           this->header_);

    // Compute cloud normals, create the normal estimation class
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud (cloud_pre_im);

    // create empty kdtree pass it to the normal estimation object based on the given input dataset
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    ne.setSearchMethod (tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    //ne.setRadiusSearch (0.01);
    ne.setKSearch(9);
    ne.compute(*cloud_normals); // Compute the features

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_RGB_norm (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::concatenateFields(*cloud_pre_im, *cloud_normals, *cloud_RGB_norm);

    // Make unorganized point cloud -> organized point cloud
    Eigen::Vector3f origin = Eigen::Vector3f(0,0,0);
    Eigen::Vector3f axis_x = Eigen::Vector3f(1,0,0);
    Eigen::Vector3f axis_y = Eigen::Vector3f(0,1,0);
    float length   = 400;
    int image_size = 500;
    float max_resolution = 2 * length / image_size;
    int max_nn_to_consider = 5;

    auto aux_cloud = ProjectToPlane(cloud_RGB_norm, origin, axis_x, axis_y);
    pcl::PointXYZRGBNormal minPt, maxPt;
    pcl::getMinMax3D (*aux_cloud, minPt, maxPt);

    auto grid = GenerateGrid(axis_x , axis_y, length, image_size, maxPt.x, minPt.x, maxPt.y, minPt.y);
    sensor_msgs::PointCloud2 cloud_proj;
    InterpolateToGrid(aux_cloud, grid, max_resolution, max_nn_to_consider);
    sensor_msgs::Image output_img;

    pcl::toROSMsg(*aux_cloud, cloud_proj);
    pcl::toROSMsg(*grid, output_img); // PointCloud2 -> image message

    cloud_proj.header.frame_id = ros_voxels.header.frame_id;
    this->pub_cloud_.publish(ros_voxels);
    //this->pub_cloud_.publish(cloud_proj);
    this->pub_indices_.publish(ros_indices);
    this->pub_img_.publish(output_img);
}

void SimpleObjectSegmentation::segmentRecursiveCC(
    UInt32Map &voxel_labels, int &label, const AdjacentList adjacency_list,
    const SupervoxelMap supervoxel_clusters, const uint32_t vindex) {
    for (auto it = adjacency_list.equal_range(vindex).first;
         it != adjacency_list.equal_range(vindex).second; ++it) {
      uint32_t n_vindex = it->second;
      auto it2 = voxel_labels.find(n_vindex);
      if (vindex != n_vindex && it2->second == -1) {
         Eigen::Vector4f sp = supervoxel_clusters.at(
            vindex)->centroid_.getVector4fMap();
         Eigen::Vector4f sn = supervoxel_clusters.at(
            vindex)->normal_.getNormalVector4fMap();
         Eigen::Vector4f np = supervoxel_clusters.at(
            n_vindex)->centroid_.getVector4fMap();
         Eigen::Vector4f nn = supervoxel_clusters.at(
            n_vindex)->normal_.getNormalVector4fMap();
         auto cc = convexityCriteria(sp, sn, np, nn, this->cc_thresh_, true);
         if (cc == 1.0f) {
            voxel_labels[n_vindex] = label;
            segmentRecursiveCC(voxel_labels, label, adjacency_list,
                               supervoxel_clusters, n_vindex);
         }
      }
    }
}

float SimpleObjectSegmentation::convexityCriteria(
    Eigen::Vector4f seed_point_, Eigen::Vector4f seed_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal,
    const float thresh, bool hard_label) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid - seed_point_).dot(n_normal);
    seed2pt_relation = (seed_point_ - n_centroid).dot(seed_normal);
    float angle = std::acos((seed_point_.dot(n_centroid)) / (
                               seed_point_.norm() * n_centroid.norm()));

    if (seed2pt_relation > thresh && pt2seed_relation > thresh) {
       float w = std::exp(-angle / (2.0f * M_PI));
       if (hard_label) {
          return 1.0f;
       } else {
          return 0.75f;
       }
    } else {
       float w = std::exp(-angle / (M_PI/6.0f));
       if (hard_label) {
          return 0.0f;
       } else {
          return 1.50f;
       }
    }
}

void SimpleObjectSegmentation::fastSeedRegionGrowing(
    PointCloudNormal::Ptr src_points, const PointCloud::Ptr cloud,
    const PointNormal::Ptr normals) {
    if (cloud->empty() || normals->size() != cloud->size()) {
       return;
    }
    int seed_index = (rect_.x + rect_.width/2)  +
       (rect_.y + rect_.height/2) * input_size_.width;
    Eigen::Vector4f seed_point_ = cloud->points[seed_index].getVector4fMap();
    Eigen::Vector4f seed_normal = normals->points[
       seed_index].getNormalVector4fMap();
    std::vector<int> labels(static_cast<int>(cloud->size()), -1);
    const int window_size = 3;
    const int wsize = window_size * window_size;
    const int lenght = std::floor(window_size/2);
    std::vector<int> processing_list;
    for (int j = -lenght; j <= lenght; j++) {
       for (int i = -lenght; i <= lenght; i++) {
          int index = (seed_index + (j * input_size_.width)) + i;
          if (index >= 0 && index < cloud->size()) {
             processing_list.push_back(index);
          }
       }
    }

    std::vector<int> temp_list;
    while (true) {
       if (processing_list.empty()) {
          break;
       }
       temp_list.clear();
       for (int i = 0; i < processing_list.size(); i++) {
          int idx = processing_list[i];
          if (labels[idx] == -1) {
             Eigen::Vector4f c = cloud->points[idx].getVector4fMap();
             Eigen::Vector4f n = normals->points[idx].getNormalVector4fMap();
             if (this->seedVoxelConvexityCriteria(
                    seed_point_, seed_normal, c, n, this->cc_thresh_) == 1) {
                labels[idx] = 1;
                for (int j = -lenght; j <= lenght; j++) {
                   for (int k = -lenght; k <= lenght; k++) {
                      int index = (idx + (j * input_size_.width)) + k;
                      if (index >= 0 && index < cloud->size()) {
                         temp_list.push_back(index);
                      }
                   }
                }
             }
          }
       }
       processing_list.clear();
       processing_list.insert(processing_list.end(), temp_list.begin(),
                              temp_list.end());
    }
    src_points->clear();
    for (int i = 0; i < labels.size(); i+=5) {
       if (labels[i] != -1) {
          PointNormalT pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          pt.r = cloud->points[i].r;
          pt.g = cloud->points[i].g;
          pt.b = cloud->points[i].b;
          pt.normal_x = normals->points[i].normal_x;
          pt.normal_y = normals->points[i].normal_y;
          pt.normal_z = normals->points[i].normal_z;
          src_points->push_back(pt);
       }
    }
}

int SimpleObjectSegmentation::seedVoxelConvexityCriteria(
    Eigen::Vector4f seed_point_, Eigen::Vector4f seed_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal, const float thresh) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid - seed_point_).dot(n_normal);
    seed2pt_relation = (seed_point_ - n_centroid).dot(seed_normal);
    if (seed2pt_relation > thresh && pt2seed_relation > thresh) {
       return 1;
    } else {
       return -1;
    }
}

int SimpleObjectSegmentation::seedVoxelConvexityCriteria(
    Eigen::Vector4f c_normal, Eigen::Vector4f n_centroid,
    Eigen::Vector4f n_normal, const float thresh) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid -
                        this->seed_point_.getVector4fMap()).dot(n_normal);
    seed2pt_relation = (this->seed_point_.getVector4fMap() - n_centroid).dot(
       this->seed_normal_.getNormalVector4fMap());
    float norm_similarity = (M_PI - std::acos(
                                c_normal.dot(n_normal) /
                                (c_normal.norm() * n_normal.norm()))) / M_PI;

    if (seed2pt_relation > thresh &&
        pt2seed_relation > thresh && norm_similarity > 0.50f) {
       return 1;
    } else {
       return -1;

    }
}

// https://stackoverflow.com/questions/49731101/generate-image-from-an-unorganized-point-cloud-in-pcl
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr SimpleObjectSegmentation::ProjectToPlane(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    Eigen::Vector3f origin, Eigen::Vector3f axis_x, Eigen::Vector3f axis_y) {

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    copyPointCloud(*cloud, *aux_cloud);

    auto normal = axis_x.cross(axis_y);
    Eigen::Hyperplane<float, 3> plane(normal, origin);

    for (auto itPoint = aux_cloud->begin(); itPoint != aux_cloud->end(); itPoint++) {
        // project point to plane
        auto proj = plane.projection(itPoint->getVector3fMap());
        itPoint->getVector3fMap() = proj;
        // optional: save the reconstruction information as normals in the projected cloud
        itPoint->getNormalVector3fMap() = itPoint->getVector3fMap() - proj;
    }
    return aux_cloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr SimpleObjectSegmentation::GenerateGrid(Eigen::Vector3f axis_x, Eigen::Vector3f axis_y,
    float length, int image_size, float max_x, float min_x, float max_y, float min_y) {

    float step_x = (max_x - min_x) / float(image_size);
    float step_y = (max_y - min_y) / float(image_size);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr image_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>(image_size, image_size));
    for (auto i = 0; i < image_size; i++) {
        for (auto j = 0; j < image_size; j++) {
            float x = i * step_x + min_x;
            float y = j * step_y + min_y;

            image_cloud->at(i, j).getVector3fMap() = (x * axis_x) + (y * axis_y);
        }
    }
    return image_cloud;
}

void SimpleObjectSegmentation::InterpolateToGrid(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr grid, float max_resolution, int max_nn_to_consider) {

    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree->setInputCloud(cloud);
    int count_zeros = 0;

    for (auto idx = 0; idx < grid->points.size(); idx++) {
        std::vector<int> indices;
        std::vector<float> distances;
        std::vector<float> colors;
        if (tree->radiusSearch(grid->points[idx], 0.002, indices, distances, 9) > 0) {
            float rgb(0);
            Eigen::Vector3f n(0, 0, 0);
            float weight_factor = 1.0F / accumulate(distances.begin(), distances.end(), 0.0F);
            /*for (auto i = 0; i < indices.size(); i++) {
                float w = weight_factor * distances[i];
                rgb += w * cloud->points[indices[i]].rgb;
                auto res = cloud->points[indices[i]].getVector3fMap() - grid->points[idx].getVector3fMap();
                n += w * res;
            }*/
            // Get most occurring color (filter out noise)
            int max_count = 0;
            for (auto i=0; i<indices.size(); i++) {
              int count = 1;
              for (auto j=i+1; j<indices.size(); j++) {
                if (cloud->points[indices[i]].rgb == cloud->points[indices[j]].rgb) {
                  count++;
                }
              }
              if (count > max_count){
                max_count = count;
              }
            }

            for (auto i=0; i<indices.size(); i++) {
              int count = 1;
              for (auto j=i+1; j<indices.size(); j++) {
                if (cloud->points[indices[i]].rgb == cloud->points[indices[j]].rgb) {
                  count++;
                }
                if (count == max_count){
                  rgb = cloud->points[indices[i]].rgb;
                }
              }
            }

            grid->points[idx].rgb = rgb;
            grid->points[idx].getNormalVector3fMap() = n;
            grid->points[idx].curvature = 1;
        }
        else {
            count_zeros++;
            grid->points[idx].rgb = 0;
            grid->points[idx].curvature = 0;
            grid->points[idx].getNormalVector3fMap() = Eigen::Vector3f(0, 0, 0);
        }
    }
    std::cout<<"num_zeros:"<< count_zeros<<std::endl;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "simple_object_segmentation");
    SimpleObjectSegmentation sos;
    ROS_INFO("SPINNING!");
    ros::spin();
    return 0;
}
