using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Core;
using Microsoft.ML.Probabilistic.Learners;
using Microsoft.ML.Probabilistic.Learners.Mappings;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using PK10CorePress;
namespace testML
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            TestML();
        }

        void TestML()
        {
            ExpectList el = new PK10ExpectReader().ReadNewestData(DateTime.Now.AddDays(-1));
            //MessageBox.Show(el.LastData.OpenCode);
            var mapping = new DaXiao_Mapping();
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);
            FeatureLabeItems trainingSet = new PKDataListSetFactory(el).OccurFeatureAndLabels();
            classifier.Train(trainingSet.FeatureVectors, trainingSet.Labels);
            List<Vector> testVector = new List<Vector>();//1
            Vector v = Vector.Zero(1);
            v[0] = 1;
            testVector.Add(v);
            var predictions = classifier.PredictDistribution(testVector);
            string estimate = classifier.Predict(0, testVector);
            MessageBox.Show(estimate);
        }

        void ProcessData()
        {
            var mapping = new JiOu_Mapping();
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);
        }

        void OccurFeaturesAndLable(ref List<Vector> Features,ref List<string> Labels)
        {

        }
    }

    /// <summary>
    /// A mapping for the Bayes Point Machine classifier tutorial.
    /// </summary>
    public class ClassifierMapping : IClassifierMapping<IList<Vector>, int, IList<string>, string, Vector>
    {
        public IEnumerable<int> GetInstances(IList<Vector> featureVectors)
        {
            for (int instance = 0; instance < featureVectors.Count; instance++)
            {
                yield return instance;
            }
        }
        public Vector GetFeatures(int instance, IList<Vector> featureVectors)
        {
            return featureVectors[instance];
        }

        public string GetLabel(int instance, IList<Vector> featureVectors, IList<string> labels)
        {
            return labels[instance];
        }

        public IEnumerable<string> GetClassLabels(IList<Vector> featureVectors = null, IList<string> labels = null)
        {
            return new[] { "Female", "Male" };
        }
    }

    public class JiOu_Mapping : IClassifierMapping<IList<Vector>, int, IList<string>, string, Vector>
    {

        public IEnumerable<int> GetInstances(IList<Vector> featureVectors)
        {
            for (int instance = 0; instance < featureVectors.Count; instance++)
            {
                yield return instance;
            }
        }
        public Vector GetFeatures(int instance, IList<Vector> featureVectors)
        {
            return featureVectors[instance];
        }

        public string GetLabel(int instance, IList<Vector> featureVectors, IList<string> labels)
        {
            return labels[instance];
        }

        public IEnumerable<string> GetClassLabels(IList<Vector> featureVectors = null, IList<string> labels = null)
        {
            return new[] { "同单","单反", "同双","双反"};
        }
    }

    public class DaXiao_Mapping : IClassifierMapping<IList<Vector>, int, IList<string>, string, Vector>
    {

        public IEnumerable<int> GetInstances(IList<Vector> featureVectors)
        {
            for (int instance = 0; instance < featureVectors.Count; instance++)
            {
                yield return instance;
            }
        }
        public Vector GetFeatures(int instance, IList<Vector> featureVectors)
        {
            return featureVectors[instance];
        }

        public string GetLabel(int instance, IList<Vector> featureVectors, IList<string> labels)
        {
            return labels[instance];
        }

        public IEnumerable<string> GetClassLabels(IList<Vector> featureVectors = null, IList<string> labels = null)
        {
            return new[] { "同大", "大反", "同小", "小反" };
        }
    }

    public class Shift_Mapping : IClassifierMapping<IList<Vector>, int, IList<string>, string, Vector>
    {

        public IEnumerable<int> GetInstances(IList<Vector> featureVectors)
        {
            for (int instance = 0; instance < featureVectors.Count; instance++)
            {
                yield return instance;
            }
        }
        public Vector GetFeatures(int instance, IList<Vector> featureVectors)
        {
            return featureVectors[instance];
        }

        public string GetLabel(int instance, IList<Vector> featureVectors, IList<string> labels)
        {
            return labels[instance];
        }

        public IEnumerable<string> GetClassLabels(IList<Vector> featureVectors = null, IList<string> labels = null)
        {
            return "-5,-4,-3,-2,-1,0,1,2,3,4,5".Split(',');
        }
    }

    public class PKDataListSetFactory
    {
        ExpectList Data;
        public PKDataListSetFactory(ExpectList el)
        {
            Data = el;
            //jisuan
        }

        public FeatureLabeItems OccurrAllProboLabels()//add by zhouys 2019/1/15
        {
            FeatureLabeItems ret = new FeatureLabeItems();
            return ret;
        }

        public FeatureLabeItems OccurFeatureAndLabels()
        {
            FeatureLabeItems ret = new FeatureLabeItems();
            for(int i=1;i<Data.Count;i++)
            {
                
                for(int col =0;col<10;col++)
                {
                    Vector v = Vector.Zero(1);
                    //string strCol = string.Format("{0}", (col + 1) % 10);
                    int CurrVal = getIntValue(Data[i].ValueList[col]);
                    int LastVal = getIntValue(Data[i-1].ValueList[col]);
                    int fVal = 0;
                    string label = "";
                    if (CurrVal <6 && LastVal<6)
                    {
                        fVal = -2;//同小
                        label = "同小";
                    }
                    if (CurrVal >= 6 && LastVal >= 6)
                    {
                        fVal = 2;//同大
                        label = "同大";
                    }
                    if(LastVal <6 && CurrVal >=6)
                    {
                        fVal = -1;//小反
                        label = "小反";
                    }
                    if (LastVal >= 6 && CurrVal < 6)
                    {
                        fVal = 1;//大反
                        label = "大反";
                    }
                    v[0]=(fVal);
                    ret.FeatureVectors.Add(v);
                    ret.Labels.Add(label);
                }
                
            }
            return ret;
        }

        int getIntValue(string val)
        {
            if (val == "0") return 10;
            return int.Parse(val);
        }
    }

    public class FeatureLabeItems
    {

        public List<Vector> FeatureVectors;
        public List<string> Labels;
        public FeatureLabeItems()
        {
            FeatureVectors = new List<Vector>();
            Labels = new List<string>();
        }
    }
}
