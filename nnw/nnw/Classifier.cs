using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AI;
using AI.ML.Classifiers;
using AI.ML.NeuralNetwork;
using AI.ML.Datasets;


namespace nnw
{
    public class Classifier : IClassifier
    {

        Net net = new Net();
        MenegerNNW mNNW;
        public int InpDim { get; private set; }
        VectorIntDataset vectorClasses = new VectorIntDataset();

        /// <summary>
        /// Классификатор
        /// </summary>
        /// <param name="inpDim">Размерность входа</param>
        public Classifier(int inpDim)
        {
            InpDim = inpDim;
            net.Add(new FullBipolyareSigmoid(InpDim, 5));
            net.Add(new Softmax( 2));
            mNNW = new MenegerNNW(net, vectorClasses);
        }


        public void Train(int epoch)
        {
            mNNW.Train(epoch);
        }

        public void AddClass(Vector[] tDataset, int nameClass)
        {
            throw new NotImplementedException();
        }

        public void AddClass(Vector data, int nameClass)
        {
            VectorClass vectorClass = new VectorClass(data, nameClass);
            vectorClasses.Add(vectorClass);
        }

        public void Open(string path)
        {
            throw new NotImplementedException();
        }

        public int Recognize(Vector inp)
        {
            return mNNW.Output(inp);
        }

        public Vector RecognizeVector(Vector inp)
        {
            throw new NotImplementedException();
        }

        public void Save(string path)
        {
            throw new NotImplementedException();
        }
    }
}
