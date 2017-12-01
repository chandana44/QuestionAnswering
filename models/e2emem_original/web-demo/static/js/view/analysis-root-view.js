define(['template/analysis-root-template', 'backbone', 'view/analysis-view-word', 'view/analysis-view'], function(analysisRootTemplate, Backbone, AnalysisWordView, AnalysisView) {

    var AnalysisRootView = Backbone.View.extend({
        initialize: function () {
            this.setElement("#analysis_container");
            this.render();
        },

        render: function () {
            var $wordView = new AnalysisWordView({ model: this.model }).render().$el,
                $analysisView = new AnalysisView({ model: this.model }).render().$el;

            this.$el.html(_.template(analysisRootTemplate));

            this.$container = this.$el.find(".analysis-view-container");
            this.$container.append($analysisView);
            this.$container.append($wordView);

            return this;
        }
    });

    return AnalysisRootView;
});
