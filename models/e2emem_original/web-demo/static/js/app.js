define(['backbone', 'model/main-model', 'view/input-view',
				 'view/result-view', 'view/analysis-root-view'], function(Backbone, Main, InputView, ResultView, AnalysisRootView) {

	app = window.app = {};

	var model = app.model = new Main();
	app.inputView = new InputView({ model: model });
	app.resultView = new ResultView({ model: model });
	app.analysisRootView = new AnalysisRootView({ model: model });

});
