define(['template/story-row-template', 'backbone'], function(storyTemplate, Backbone) {

   var  StoryRowView = Backbone.View.extend({
        initialize: function(options) {
            this.story = options.story;
            this.parent = options.parent;
        },

        events: {
            'click .btn-primary': 'onStorySelect'
        },

        render: function() {
            var template = _.template(storyTemplate, {
                story: this.story.s,
                question: this.story.q,
                answer: this.story.a,
                task: this.story.t
            });

            this.$el.html(template);

            return this;
        },

        onStorySelect: function () {
            this.model.set({
                "answer": "",
                "correctAnswer": this.story.a,
                "answerProbability": 0,
                "memoryProbabilities": []
            });

            app.inputView.setValues(this.story.s, this.story.q);

            $('#myModal').modal('hide');
        }
    });

    return StoryRowView;
});
