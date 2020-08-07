//--------------------------------C-Code

#include "./include/System.h"
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <opencv2/core/core.hpp>

#include <string>
#include "lib/conversion.h"

namespace bpy = boost::python;
using namespace std;

//  need to initialize numpy
#if PY_MAJOR_VERSION >= 3
int
#else
void
#endif
init_numpy()
{
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();
}


class SLAMClass{
    public:
        SLAMClass();

        NDArrayConverter cvt;

        void init(const std::string &path_to_vocabulary, const std::string &path_to_settings, const string camera_type, const bool flag_visualize){
            if (camera_type == "rgbd") {
                this->slam_obj = new ORB_SLAM2::System(path_to_vocabulary, path_to_settings, ORB_SLAM2::System::RGBD, flag_visualize);
            } else if (camera_type == "stereo") {
                this->slam_obj = new ORB_SLAM2::System(path_to_vocabulary, path_to_settings, ORB_SLAM2::System::STEREO,
                                                       flag_visualize);
            } else {
                this->slam_obj = new ORB_SLAM2::System(path_to_vocabulary, path_to_settings, ORB_SLAM2::System::MONOCULAR, flag_visualize);
            }
            std::cout << path_to_vocabulary << std::endl;
        }

        PyObject* track_rgbd(PyObject* rgb_img, PyObject* d_img, const double timestamp){
            cv::Mat rgb_img_mat = cvt.toMat(rgb_img);
            cv::Mat d_img_mat = cvt.toMat(d_img);
            auto camera_matrix = this->slam_obj->TrackRGBD(rgb_img_mat, d_img_mat, timestamp);
            PyObject* ret = cvt.toNDArray(camera_matrix);
            return ret;
        }

        PyObject* visualize_cape(PyObject* rgb_img, PyObject* d_img){
            cv::Mat rgb_img_mat = cvt.toMat(rgb_img);
            cv::Mat d_img_mat = cvt.toMat(d_img);
            return cvt.toNDArray(this->slam_obj->VisualizeCape(rgb_img_mat, d_img_mat));;
        }

        PyObject* track_stereo(PyObject* imgl, PyObject* imgr, const double timestamp) {
            cv::Mat imgl_mat = cvt.toMat(imgl);
            cv::Mat imgr_mat = cvt.toMat(imgr);
            auto camera_matrix = this->slam_obj->TrackStereo(imgl_mat, imgr_mat, timestamp);
            PyObject * ret = cvt.toNDArray(camera_matrix);
            return ret;
        }
        PyObject* track_mono(PyObject* img, const double timestamp){
            cv::Mat img_mat = cvt.toMat(img);
            auto camera_matrix = this->slam_obj->TrackMonocular(img_mat, timestamp);
            PyObject* ret = cvt.toNDArray(camera_matrix);
            return ret;
        }

        PyObject* getmap(){
            return cvt.toNDArray(this->slam_obj->GetMapCloud());
        }

        void deactivate_mapping() {
            this->slam_obj->ActivateLocalizationMode();
        }

        void activate_mapping() {
            this->slam_obj->DeactivateLocalizationMode();
        }

        void shutdown() {
            this->slam_obj->Shutdown();
        }

        PyObject* get_feature_kps(){
            cv::Mat out;
            cv::KeyPoint kp;
            for(auto kp: this->slam_obj->GetTrackedKeyPointsUn()){
                out.push_back(kp.pt);
            }
            return cvt.toNDArray(out);
        }

        void prepare_dump(){
            slam_obj->PrepareDump();
        }
        PyObject* get_kf_ids_from_mps(){
            return cvt.toNDArray(slam_obj->dump_kf_ids_from_mps());
        }
        PyObject* get_kf_ids(){
            return cvt.toNDArray(slam_obj->dump_kf_ids());
        }
        PyObject* get_mp_3dpts(){
            return cvt.toNDArray(slam_obj->dump_mp_3dpts());
        }
        PyObject* get_kf_3dpts(){
            return cvt.toNDArray(slam_obj->dump_kf_3dpts());
        }

        PyObject* get_kf_ids_from_planes(){
            return cvt.toNDArray(slam_obj->dump_kf_ids_from_planes());
        }
        PyObject* get_plane_params(){
            return cvt.toNDArray(slam_obj->dump_plane_params());
        }
        PyObject* get_frame_ids(){
            return cvt.toNDArray(slam_obj->dump_frame_ids());
        }
        PyObject* get_plane_segs(){
            return cvt.toNDArray(slam_obj->dump_plane_segs());
        }
        PyObject* get_kf_clouds(){
            return cvt.toNDArray(slam_obj->dump_kf_clouds());
        }
//    SYSTEM_NOT_READY=-1,
//            NO_IMAGES_YET=0,
//            NOT_INITIALIZED=1,
//            OK=2,
//            LOST=3

    int tracking_state() {
        return this->slam_obj->GetTrackingState();
    }

        ~SLAMClass();


    private:
        ORB_SLAM2::System *slam_obj;
};

    SLAMClass::SLAMClass() {
        // nothing
    }

    SLAMClass::~SLAMClass() {
        delete this->slam_obj; // Otherwise may end up with a SegFault!!
    }

//--------------------------------Boost-Python-Code


BOOST_PYTHON_MODULE(ORBSLAM2)
{
    using namespace boost::python;
    Py_Initialize(); // required by NumPy
    init_numpy(); // required by NumPy

    class_<SLAMClass>("SLAM")
        .def("init", &SLAMClass::init)
        .def("track_rgbd", &SLAMClass::track_rgbd)
        .def("track_stereo", &SLAMClass::track_stereo)
        .def("track_mono", &SLAMClass::track_mono)
        .def("getmap", &SLAMClass::getmap)
        .def("deactivate_mapping", &SLAMClass::deactivate_mapping)
        .def("shutdown", &SLAMClass::shutdown)
        .def("tracking_state", &SLAMClass::tracking_state)
        .def("get_feature_kps", &SLAMClass::get_feature_kps)
        .def("prepare_dump", &SLAMClass::prepare_dump)
        .def("get_kf_ids_from_mps", &SLAMClass::get_kf_ids_from_mps)
        .def("get_kf_ids", &SLAMClass::get_kf_ids)
        .def("get_mp_3dpts", &SLAMClass::get_mp_3dpts)
        .def("get_kf_3dpts", &SLAMClass::get_kf_3dpts)
        .def("visualize_cape", &SLAMClass::visualize_cape)
        .def("get_kf_ids_from_planes", &SLAMClass::get_kf_ids_from_planes)
        .def("get_plane_params", &SLAMClass::get_plane_params)
        .def("get_frame_ids", &SLAMClass::get_frame_ids)
        .def("get_plane_segs", &SLAMClass::get_plane_segs)
        .def("get_kf_clouds", &SLAMClass::get_kf_clouds)
        .def("activate_mapping", &SLAMClass::activate_mapping);
}
