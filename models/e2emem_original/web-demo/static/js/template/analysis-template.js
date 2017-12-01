define([], function() {

    return [
        '<div class="form-group jumbotron">' +
            '<table class="hop-table table">' +

            '</table>' +
        '</div>',

        '<tr>' +
            '<td><%= sentence %></td>' +
            '<td style="background-color:rgba(102, 179, 255, <%= perc1 %> )"><%= perc1 %></td>' +
            '<td style="background-color:rgba(102, 179, 255, <%= perc2 %> )"><%= perc2 %></td>' +
            '<td style="background-color:rgba(102, 179, 255, <%= perc3 %> )"><%= perc3 %></td>' +
        '</tr>',

        '<tr>' +
            '<th>Sentence</th>' +
            '<th>Hop 1</th>' +
            '<th>Hop 2</th>' +
            '<th>Hop 3</th>' +
        '</tr>'
        ];

});
