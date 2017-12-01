define(['template/analysis-word-template', 'backbone', 'jquery-ui'], function(analysisWordTemplate, Backbone, slider) {

    var MAX_SENTENCE_LENGTH = 11,
      MAX_HOPS = 3,
      WORD_IMPORTANCE_VECTOR = [ 0.0158531 ,  0.01214825,  0.01591476,  0.21010481,  0.0142038 ,
        0.14066565,  0.16964697,  0.1430412 ,  0.09100302,  0.1546434 ,
        0.03277504];

    var AnalysisWordView = Backbone.View.extend({
        initialize: function () {
            //this.setElement("#analysis_word_container");
            //this.render();

            this.listenTo(this.model, "change:answer", this._onAnswer);
            this.listenTo(this.model, "change:hop", this._onHop);
        },

        render: function () {
            var template = _.template(analysisWordTemplate[0]),
              self = this;
            this.$el.html(template);

            this.$table = this.$el.find(".table");
            this.$slider = this.$el.find(".hop-slider");

            this.$slider.slider({
              min: 0,
              value: 0,
              max: MAX_HOPS - 1,
              change: function (event, ui) {
                self.model.set("hop", ui.value);
              }
            });

            this.renderContent();

            return this;
        },

        renderContent: function () {
            this.renderSentences();

            this.$slider.slider({
              value: 0
            });
        },

        renderSentences: function () {
            var sentences = this.model.get('story'),
                hop = this.model.get('hop'),
                self = this;

            var probs = this.model.get('memoryProbabilities');

            if (probs.length === 0)
                return;

          _.each(sentences, function(val, idx) {
            var words = val.split(' '),
              prob = probs[idx],
              hopProb = prob[hop],
              cellTemplate = analysisWordTemplate[1],
              $row = $('<tr style="background-color:rgba(255, 158, 158, ' + hopProb + ' )"></tr>');

            $row.append(_.template(cellTemplate, {'word': hopProb.toFixed(3), 'weight': 0} ));

            for (var i=0; i<MAX_SENTENCE_LENGTH; i++) {
              var w = '';

              if (i >= words.length)
                w = '';
              else
                w = words[i];

              $row.append(_.template(cellTemplate, {'word': w, 'weight': 1 * WORD_IMPORTANCE_VECTOR[i]} ));
            }

            self.$table.append($row);
          });

        },

        _onHop: function () {
            this.$table.empty();
            this.renderSentences();
        },

        _onAnswer: function () {
            this.$table.empty();
            this.renderContent();
        }
    });

    return AnalysisWordView;
});
