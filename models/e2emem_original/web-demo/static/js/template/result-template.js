define([], function() {

    return '<div class="col-md-3">' +
        '<div class="form-group jumbotron">' +
            '<h3>Answer</h3>' +
            '<h4><span class="label label-default"><%= answer %></span></h4>' +
            '<h3>Confidence</h3>' +
            '<div class="progress">' +
              '<div class="progress-bar <%= progressClass %>" role="progressbar" aria-valuenow="<%= confidence %>" aria-valuemin="0" aria-valuemax="100" style="width: <%= confidence %>%;">' +
                '<%= confidence %>%' +
              '</div>' +
            '</div>' +
            '<h3 class="<%= correctAnswerShown %>">Correct answer</h3>' +
            '<h4 class="<%= correctAnswerShown %>"><span class="label label-default"><%= correctAnswer %></span></h4>' +
        '</div>' +

    '</div>';

});
