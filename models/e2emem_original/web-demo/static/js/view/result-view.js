define(['template/result-template', 'backbone'], function(resultTemplate, Backbone) {

    var ResultView = Backbone.View.extend({
        initialize: function() {
            this.setElement("#result_container");
            this.render();

            this.listenTo(this.model, "change:answer", this._onAnswer);
        },

        render: function() {
            this.renderContent();

            return this;
        },

        renderContent: function () {
            var confidence = (this.model.get('answerProbability') * 100).toFixed(1),
                correctAnswer = this.model.get("correctAnswer"),
                correctAnswerShown = !!correctAnswer,
                progressClass = "";

            if (confidence > 80)
                progressClass = "progress-bar-success";
            else if (confidence > 50)
                progressClass = "progress-bar-warning";
            else
                progressClass = "progress-bar-danger";

            var template = _.template(resultTemplate, {
                "answer": this.model.get("answer"),
                "correctAnswer": correctAnswer,
                "correctAnswerShown": correctAnswerShown ? '' : 'hidden',
                "confidence": confidence,
                "progressClass": progressClass
            });
            this.$el.html(template);
        },

        _onAnswer: function () {
            this.$el.empty();
            this.renderContent();
        }
    });

    return ResultView;
});
