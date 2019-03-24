using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using AI;
using AI.ML.Datasets;
using AI.Signals;
using AI.Statistics;
using AI.AlgorithmAnalysis;

namespace nnw
{
    public partial class Form1 : Form
    {

        Classifier classifier;
        Vector t = Vector.Time0(500, 1); // Отсчеты времени частота дискретизации = 500 Гц, время реализации =  1 сек
        VectorIntDataset vectorClassesTest = new VectorIntDataset();
        Random random = new Random();
        double k = 2.5; // коэффициент шума



        public Form1()
        {
            InitializeComponent();
            classifier = new Classifier(t.N);
        }

        Vector GetSin()
        {
            return Signal.Sin(t, 4) + k*Statistic.randNorm(t.N, random);
        }

        Vector GetRect()
        {
            return 2*Signal.Rect(t, 4)-1 + k*Statistic.randNorm(t.N, random); // Превидение к тому масштабу что и синус, что бы распознавание не происходило только по мат. ожиданию
        }


        private void Form1_Load(object sender, EventArgs e)
        {
            for (int i = 0; i < 1000; i++)
            {
                classifier.AddClass(GetSin(), 0);
                classifier.AddClass(GetRect(), 1);

                vectorClassesTest.Add(new VectorClass(GetSin(), 0));
                vectorClassesTest.Add(new VectorClass(GetRect(), 1));
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            chartControl1.Clear();
            var sin = GetSin();
            chartControl1.AddPlot(t, sin, "Синус", Color.Green, 2);

            label1.Text = classifier.Recognize(sin) == 0 ? "Синус" : "Прямоугольный";
        }

        private void button2_Click(object sender, EventArgs e)
        {
            chartControl1.Clear();
            var rect = GetRect();
            chartControl1.AddPlot(t, rect, "Прямоугольный", Color.Green, 2);

            label1.Text = classifier.Recognize(rect) == 0 ? "Синус" : "Прямоугольный";
        }

        private void button4_Click(object sender, EventArgs e)
        {
            MessageBox.Show((Metrics.Pressicion(classifier, vectorClassesTest, 2)*100).ToString(),"Точность");
        }

        private void button3_Click(object sender, EventArgs e)
        {
            classifier.Train(2);
        }
    }
}
