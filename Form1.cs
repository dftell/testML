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
            //ExpectList el = new PK10ExpectReader().ReadNewestData(725888,1200);//725888以前1200
            ExpectList el = new PK10ExpectReader().ReadHistory()
            //MessageBox.Show(el.LastData.OpenCode);
            var mapping = new Serial_Mapping();
            DataTable dtAll = new DataTable();
            Dictionary<int, string> ret = new Dictionary<int, string>();
            PKDataListSetFactory pksf = new PKDataListSetFactory(el);
            Dictionary<int,int> res =  pksf.OccurProbList(1000, 1);
            MessageBox.Show(string.Join(";", res.Values.ToList()));
            ////////////////for (int i=0;i<10;i++)
            ////////////////{
            ////////////////    dtAll.Columns.Add(string.Format("key{0}", i), typeof(string));
            ////////////////    dtAll.Columns.Add(string.Format("val{0}", i), typeof(double));
            ////////////////}
            ////////////////for(int i=0;i<10;i++)
            ////////////////{
            ////////////////    dtAll.Rows.Add(dtAll.NewRow());
            ////////////////}
            ////////////////StringBuilder sb = new StringBuilder();//删除packages
            ////////////////for (int i = 0; i < 10; i++)
            ////////////////{
                
            ////////////////    var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);
            ////////////////    FeatureLabeItems trainingSet = pksf.OccurColumnFeatureAndLabels(i, 1000, 1);
            ////////////////    classifier.Train(trainingSet.FeatureVectors, trainingSet.Labels);
            ////////////////    List<Vector> testVector = new List<Vector>();//1
            ////////////////    Vector v = Vector.Zero(1);
            ////////////////    v[0] = int.Parse(el.LastData.ValueList[i]=="0"?"10":el.LastData.ValueList[i]);
            ////////////////    testVector.Add(v);
            ////////////////    var predictions = classifier.PredictDistribution(testVector);
            ////////////////    string estimate = classifier.Predict(0, testVector);
            ////////////////    sb.AppendLine(estimate);
            ////////////////    int row1 = 0;
            ////////////////    foreach(var vv in predictions)
            ////////////////    {
            ////////////////        foreach(string key in vv.Keys)
            ////////////////        {
            ////////////////            dtAll.Rows[row1][string.Format("key{0}", i)] = key;
            ////////////////            dtAll.Rows[row1][string.Format("val{0}", i)] = vv[key];
            ////////////////            row1++;
            ////////////////        }
            ////////////////    }
            ////////////////}
            ////////////////DataTable mydt = new DataTable();
            ////////////////for (int i = 0; i < 10; i++)
            ////////////////{
            ////////////////    mydt.Columns.Add(string.Format("key{0}", i), typeof(string));
            ////////////////    mydt.Columns.Add(string.Format("val{0}", i), typeof(double));
            ////////////////}
            ////////////////for (int i = 0; i < 101; i++)
            ////////////////{
            ////////////////    mydt.Rows.Add(mydt.NewRow());
            ////////////////}
            ////////////////for (int i = 0; i < 10; i++)
            ////////////////{
            ////////////////    Dictionary<string, double> mydata = pksf.OccurColumnProb(i, 1000, 1);
            ////////////////}
            ////////////////int row = 0;
            ////////////////double sum = 0; 
            ////////////////foreach(string key in mydata.Keys)
            ////////////////{
            ////////////////    mydt.Rows[row][string.Format("key{0}", 0)] = key;
            ////////////////    mydt.Rows[row][string.Format("val{0}", 0)] = mydata[key];
            ////////////////    sum += mydata[key];
            ////////////////    row++;
            ////////////////}
            ////////////////DataView dv = new DataView(dtAll);
            ////////////////mydt.Rows[100][string.Format("key{0}", 0)] = "合计";
            ////////////////mydt.Rows[100][string.Format("val{0}", 0)] = sum;
            ////////////////this.dataGridView1.DataSource = mydt;
            ////////////////MessageBox.Show(BayesDicClass.getBAMaxValue(mydata, 5).ToString());
            //MessageBox.Show(sb.ToString());

        }

        int CheckMatchCnt(Dictionary<int,int> res,string strNo)
        {
            return 0;
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

    public class PKDataListSetFactory
    {
        ExpectList Data;
        public PKDataListSetFactory(ExpectList el)
        {
            Data = el;
            //jisuan
        }

        BayesDicClass OccurrDir(int col, int TestLength, int LastTimes)//add by zhouys 2019/1/15
        {
            BayesDicClass ret = new BayesDicClass();
            int iShift = Data.Count - TestLength;
            if (iShift <= LastTimes) //Data length must more than TestLength+LastTimes+1
                return ret;
            Dictionary<string, int> defaultDic = getDefaultCombDic();
            Dictionary<int, int> PreA = InitPriorProbabilityDic();
            Dictionary<int, int> PreB = InitPriorProbabilityDic();
            //for (int col=0;col<10;col++)
            //{
            Dictionary<string, int> combDic = defaultDic;
            for (int i = iShift - 1; i < Data.Count; i++)
            {
                int CurrA = getIntValue(Data[i].ValueList[col]);
                int CurrB = getIntValue(Data[i - LastTimes].ValueList[col]);
                string key = string.Format("{0}_{1}", CurrA, CurrB);
                int cnt = combDic[key];
                combDic[key] = cnt + 1;
                PreA[CurrA] = PreA[CurrA] + 1;
                PreB[CurrB] = PreB[CurrB] + 1;
            }
            ret.PosteriorProbDic = combDic;
            ret.PriorProbDicA = PreA;
            ret.PriorProbDicB = PreB;
            ret.TestLength = TestLength;
            //}
            return ret;
        }

        Dictionary<int, int> InitPriorProbabilityDic()
        {
            Dictionary<int, int> ret = new Dictionary<int, int>();
            for (int i = 1; i <= 10; i++)
                ret.Add(i % 10, 0);
            return ret;
        }




        Dictionary<string, int> getDefaultCombDic()
        {
            Dictionary<string, int> ret = new Dictionary<string, int>();
            for (int i = 1; i <= 10; i++)
            {
                for (int j = 1; j <= 10; j++)
                {
                    string key = string.Format("{0}_{1}", i % 10, j % 10);
                    ret.Add(key, 0);
                }
            }
            return ret;
        }

        public Dictionary<string, double> OccurColumnProb(int col, int TestLength, int LastTimes)
        {
            Dictionary<string, double> ret = new Dictionary<string, double>();
            BayesDicClass bdic = OccurrDir(col, TestLength, LastTimes);
            ret = bdic.getBA();
            return ret;
        }

        public Dictionary<int, List<int>> OccurProbList(int TestLength, int LastTimes, int SelectListCnt)
        {
            Dictionary<int, List<int>> ret = new Dictionary<int, List<int>>();
            for (int i = 0; i < 10; i++)
            {
                Dictionary<string, double> res = OccurColumnProb(i, TestLength, LastTimes);
                string str = Data.LastData.ValueList[i];
                //str = str == "0" ? "10" : str;
                int col = (i + 1) % 10;
                List<int> colList = BayesDicClass.getBAMaxNValue(res, int.Parse(str), SelectListCnt);
                ret.Add(col, colList);
            }
            return ret;
        }

        public Dictionary<int, int> OccurProbList(int TestLength, int LastTimes)
        {
            Dictionary<int, List<int>> ret = OccurProbList(TestLength, LastTimes, 1);
            return ret.ToDictionary(p => p.Key, p => p.Value[0]);
        }




        int getIntValue(string val)
        {
            //if (val == "0") return 10;
            return int.Parse(val);
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

    public class BayesDicClass
    {
        public int TestLength;
        public Dictionary<string, int> PosteriorProbDic;
        public Dictionary<int, int> PriorProbDicA;
        public Dictionary<int, int> PriorProbDicB;

        public Dictionary<string,double> getBA()
        {
            Dictionary<string, double> ret = new Dictionary<string, double>();
            foreach(string key in PosteriorProbDic.Keys)
            {
                string[] strIdx = key.Split('_');
                int A = int.Parse(strIdx[0]);
                int B = int.Parse(strIdx[1]);
                double ProbBA = (double)PriorProbDicA[A] * PosteriorProbDic[key] / PriorProbDicB[B] / TestLength;
                ret.Add(key, ProbBA);
            }
            return ret;
        }

        public static Dictionary<int,double> getBADic(Dictionary<string,double> testResult, int CheckVal)
        {
            Dictionary<int, double> ret = new Dictionary<int, double>();
            for (int i=1;i<=10;i++)
            {
                string key = string.Format("{0}_{1}",i%10,CheckVal);
                ret.Add(i%10, testResult[key]);
            }
            return ret;
        }

        public static string getBAString(Dictionary<string, double> testResult, int CheckVal)
        {
            Dictionary<int, double> ret = getBADic(testResult, CheckVal);
            ret.OrderBy(p => p.Value);
            StringBuilder sb = new StringBuilder();
            double sum = 0;
            foreach (int key in ret.Keys)
            {
                sb.AppendLine(string.Format("{0}:{1}", key, ret[key]));
                sum += ret[key];
            }
            sb.AppendLine(string.Format("合计:{0}", sum));
            return sb.ToString();
        }

        public static int getBAMaxValue(Dictionary<string, double> testResult, int CheckVal)
        {
            Dictionary<int, double> ret = getBADic(testResult, CheckVal);
            return ret.OrderByDescending(p => p.Value).First().Key;
            //return ret.First().Key;
        }

        public static List<int> getBAMaxNValue(Dictionary<string, double> testResult, int CheckVal,int MaxN)
        {
            Dictionary<int, double> ret = getBADic(testResult, CheckVal);
            var list = ret.OrderByDescending(p => p.Value);
            return list.ToDictionary(p => p.Key, p => p.Value).Keys.Take(MaxN).ToList();
            //.Take(MaxN).ToList();
            //return ret.First().Key;
        }
    }
}
