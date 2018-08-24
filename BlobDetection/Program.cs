using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace BlobDetection
{
    class Program
    {
        private static string _input = "blob.jpg";

        static void Main()
        {
            var image = CvInvoke.Imread(_input, ImreadModes.Grayscale);

            var parameters = new SimpleBlobDetectorParams
            {
                MinThreshold = 10,
                MaxThreshold = 200,

                FilterByArea = true,
                MinArea = 1500,

                FilterByCircularity = true,
                MinCircularity = 0.1f,

                FilterByConvexity = true,
                MinConvexity = 0.87f,

                FilterByInertia = true,
                MinInertiaRatio = 0.01f
            };

            var detector = new SimpleBlobDetector(parameters);
            var keypoints = new VectorOfKeyPoint(detector.Detect(image));
            var output = new Mat();
            Features2DToolbox.DrawKeypoints(image, keypoints, output, new Bgr(0, 0, 255), Features2DToolbox.KeypointDrawType.DrawRichKeypoints);
            CvInvoke.Imshow("keypoints", output);
            CvInvoke.WaitKey();
        }
    }
}