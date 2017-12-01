define(["backbone"], function(Backbone) {
    var Main = Backbone.Model.extend({

        defaults: {
            "hop": 0,
            "story": [],
            "question": "",
            "answer": "",
            "correctAnswer": "",
            "answerProbability": 0,
            "memoryProbabilities": []
        },

        getAnswer: function () {
            var data = {
                "sentences": this.get("story"),
                "question": this.get("question")
            };

            $.post('answer', {
                'data': JSON.stringify(data)
            })
            .then(_.bind(function (resp) {
                this.set("hop", 0, {"silent": true});

                this.set({
                    "answer": resp.answer,
                    "answerProbability": resp.answerProbability,
                    "memoryProbabilities": resp.memoryProbabilities
                });

                this.trigger('change:answer');
            }, this));
        }

    });

    return Main;
});
