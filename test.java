package top.mk.ms.djl;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.SampleForecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;
import ai.djl.timeseries.evaluator.Rmsse;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.translator.DeepARTranslator;
import ai.djl.timeseries.translator.DeepARTranslatorFactory;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import com.google.gson.GsonBuilder;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.noear.snack.ONode;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.*;

public class test {
    public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
       //查看所有模型
//        Map<Application, List<Artifact>> applicationListMap = ModelZoo.listModels();

        //模型过滤器
        Criteria<TimeSeriesData, Forecast> criteria =
                Criteria.builder()
                        //选择模型
                        .optApplication(Application.TimeSeries.FORECASTING)
                        //输入输出参数
                        .setTypes(TimeSeriesData.class, Forecast.class)
                        //选择引擎
                        .optEngine("PyTorch")
                        //翻译器（可以自定义）
                        .optTranslatorFactory(new DeepARTranslatorFactory())
                        //翻译器 参数设置
                        .optArgument("prediction_length", 4)
                        .optArgument("freq", "W")
                        //是否使用什么动态
                        .optArgument("use_feat_dynamic_real", "false")
                        //  //是否使用什么静态分类
                        .optArgument("use_feat_static_cat", "false")
                        //是否使用什么静态
                        .optArgument("use_feat_static_real", "false")
                        //指定模型加载进度(控制台打印进度)
                        .optProgress(new ProgressBar())
                        .build();

        //加载模型
        ZooModel<TimeSeriesData, Forecast> model = criteria.loadModel();


        Engine engine = Engine.getInstance();
        NDManager manager = engine.newBaseManager();
        //获取模型自带测试数据
//        M5Dataset dataset = M5Dataset.builder().setManager(manager).build();
        TimeSeriesData input = getTimeSeriesData(manager);
        //训练
        DistributionOutput distributionOutput = new NegativeBinomialOutput();
        DefaultTrainingConfig config = setupTrainingConfig(distributionOutput);
        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());


//        EasyTrain.fit(trainer, 5, null, null);

        //创建预测器
//        Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor();
//        //预测
//        NDArray target = input.get(FieldName.TARGET);
//        target.setName("target");
////        saveNDArray(target);
//
//        Forecast forecast = predictor.predict(input);
//
//        float[] floatArray = forecast.mean().toFloatArray();
//        System.out.println(ONode.load(floatArray).toJson());

        manager.close();


        System.out.println(1);
    }
    private static TimeSeriesData getTimeSeriesData(NDManager manager)   {
        TimeSeriesData data = new TimeSeriesData(10);
        //开始时间
        data.setStartTime(LocalDateTime.parse("2011-01-29T00:00"));
        NDArray target = manager.create(new float[]{  3.860f,3.390f, 4.050f,   3.510f,  3.160f,4.160f, 3.540f, 3.230f,3.670f, 3.240f, 3.900f, 3.840f,3.370f});
        data.setField(FieldName.TARGET, target);
        return data;

    }
    private static DefaultTrainingConfig setupTrainingConfig( DistributionOutput distributionOutput) {
        String outputDir = "build/model";
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float rmsse = result.getValidateEvaluation("RMSSE");
                    model.setProperty("RMSSE", String.format("%.5f", rmsse));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new DistributionLoss("Loss", distributionOutput))
                .addEvaluator(new Rmsse(distributionOutput))
                .optDevices(Engine.getInstance().getDevices(Engine.getInstance().getGpuCount()))
                .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT)
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }
}
