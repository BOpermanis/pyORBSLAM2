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

//    PyObject* getmap(){
////            auto mappoints = this->slam_obj->GetAllMapPoints();
//            auto mappoints = this->slam_obj->GetTrackedMapPoints();
//            int n = mappoints.size();
//            cv::Mat out;
//            for(int i=0; i<n; i++){
//                out.push_back(mappoints[i]->GetWorldPos());
//                auto v = mappoints[i]->GetReferenceKeyFrame2();
//                out.push_back(v);
//            }
//            return cvt.toNDArray(out);
//        }

    PyObject* getmap(){
        return cvt.toNDArray(this->slam_obj->GetMapCloud());
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
        ;
}