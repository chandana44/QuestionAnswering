define(['template/story-loader-template', 'backbone', 'demo-data/demo-stories', 'view/story-row-view'], function(storyTemplate, Backbone, stories, StoryRow) {

   var  StoryLoaderView = Backbone.View.extend({
        initialize: function() {
            this.render();
        },

        render: function() {
            var template = _.template( storyTemplate);
            this.$el.html(template);

            this.$modalBody = this.$el.find('.modal-body');

            this._showStories();

            return this;
        },

        _showStories: function () {
            var self = this;

            this.$modalBody.empty();

            _.each(stories, function (story) {
                var story = new StoryRow({
                  model: self.model,
                  story: story
                }).render();

                self.$modalBody.append(story.$el);
            });
        }
    });

    return StoryLoaderView;
});
