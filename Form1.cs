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
            //TestML();
            TestSerial();
        }

        void TestML()
        {
            ExpectList el = new PK10ExpectReader().ReadNewestData(DateTime.Now.AddDays(-17));//至少180*16天+当天的记录数>1000
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

        void TestSerial()
        {
            //ExpectList el = new PK10ExpectReader().ReadNewestData(DateTime.Now.AddDays(-17));//至少180*16天+当天的记录数>1000
            ExpectList el = new PK10ExpectReader().ReadNewestData(725888,1200);//725888以前1200
            //MessageBox.Show(el.LastData.OpenCode);
            var mapping = new Serial_Mapping();
            DataTable dtAll = new DataTable();
            Dictionary<int, string> ret = new Dictionary<int, string>();
            for(int i=0;i<10;i++)
            {
                dtAll.Columns.Add(string.Format("key{0}", i), typeof(string));
                dtAll.Columns.Add(string.Format("val{0}", i), typeof(double));
            }
            for(int i=0;i<10;i++)
            {
                dtAll.Rows.Add(dtAll.NewRow());
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 10; i++)
            {
                var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);
                FeatureLabeItems trainingSet = new PKDataListSetFactory(el).OccurColumnFeatureAndLabels(i, 1000, 1);
                classifier.Train(trainingSet.FeatureVectors, trainingSet.Labels);
                List<Vector> testVector = new List<Vector>();//1
                Vector v = Vector.Zero(1);
                v[0] = int.Parse(el.LastData.ValueList[i]=="0"?"10":el.LastData.ValueList[i]);
                testVector.Add(v);
                var predictions = classifier.PredictDistribution(testVector);
                string estimate = classifier.Predict(0, testVector);
                sb.AppendLine(estimate);
                int row = 0;
                foreach(var vv in predictions)
                {
                    foreach(string key in vv.Keys)
                    {
                        dtAll.Rows[row][string.Format("key{0}", i)] = key;
                        dtAll.Rows[row][string.Format("val{0}", i)] = vv[key];
                        row++;
                    }
                }
            }
            
            DataView dv = new DataView(dtAll);

            this.dataGridView1.DataSource = dv;
            MessageBox.Show(sb.ToString());

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

    public class Serial_Mapping : IClassifierMapping<IList<Vector>, int, IList<string>, string, Vector>
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
            //////string[] ret = new string[100];
            //////for (int i = 1; i <= 10; i++)
            //////    for (int j = 1; j <= 10; j++)
            //////        ret[(i-1) * 10 + (j-1)] = string.Format("{0}_{1}", i, j);
            //////return ret;
            return "1,2,3,4,5,6,7,8,9,10".Split(',');
            //return new[] { "同大", "大反", "同小", "小反" };
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

        Dictionary<int,Dictionary<string,int>> OccurrDir(int TestLength,int LastTimes)//add by zhouys 2019/1/15
        {
            Dictionary<int, Dictionary<string, int>> ret = new Dictionary<int, Dictionary<string, int>>();
            
            int iShift = Data.Count - TestLength;
            if (iShift <= LastTimes) //Data length must more than TestLength+LastTimes+1
                return ret;
            Dictionary<string, int> defaultDic = getDefaultCombDic();
            for (int col=0;col<10;col++)
            {
                Dictionary<string, int> combDic = defaultDic;
                for (int i = iShift-1; i < Data.Count; i++)
                {
                    int CurrVal = getIntValue(Data[i].ValueList[col]);
                    int LastVal = getIntValue(Data[i - LastTimes].ValueList[col]);
                    string key = string.Format("{0}_{1}", CurrVal, LastVal);
                    int cnt = combDic[key];
                    combDic[key] = cnt + 1;
                }
                ret.Add(col + 1, combDic);
            }
            return ret;
        }



        Vector getVectorByProboDic(int TestLength,Dictionary<string,int> dic)
        {
            Vector ret = Vector.Zero(100);
            int i = 0;
            foreach(string key in dic.Keys)
            {
                ret[i] = (double)dic[key]/TestLength;
                i++;
            }
            return ret;
        }

        
        Dictionary<string,int> getDefaultCombDic()
        {
            Dictionary<string, int> ret = new Dictionary<string, int>();
            for(int i=1;i<=10;i++)
            {
                for(int j=1;j<=10;j++)
                {
                    string key = string.Format("{0}_{1}", i, j);
                    ret.Add(key, 0);
                }
            }
            return ret;
        }

        public FeatureLabeItems OccurColumnFeatureAndLabels(int col,int TestLength, int LastTimes)
        {
            FeatureLabeItems ret = new FeatureLabeItems();
            int iShift = Data.Count - TestLength;
            if (iShift <= LastTimes) //Data length must more than TestLength+LastTimes+1
                return ret;
            Dictionary<string, int> defaultDic = getDefaultCombDic();

            string BaseLabe = "";// string.Format("前{1}期",col+1, LastTimes);
            Dictionary<string, int> combDic = defaultDic;
            for (int i = iShift - 1; i < Data.Count; i++)
            {
                int CurrVal = getIntValue(Data[i].ValueList[col]);
                int LastVal = getIntValue(Data[i - LastTimes].ValueList[col]);
                string key = string.Format("{0}_{1}", CurrVal,LastVal);
                key = string.Format("{0}", CurrVal);
                Vector v = Vector.Zero(1);
                v[0] = LastVal;
                ret.FeatureVectors.Add(v);
                ret.Labels.Add(key);
                //int cnt = combDic[key];
                //combDic[key] = cnt + 1;
            }
            //ret.Add(col + 1, combDic);
           
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
