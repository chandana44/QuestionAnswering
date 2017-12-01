define([], function() {

    return '<button type="button" class="btn btn-warning" data-toggle="modal" data-target="#myModal">Load</button>' +

            '<!-- Modal -->' +
            '<div id="myModal" class="modal fade" role="dialog">' +
              '<div class="modal-dialog modal-lg">' +

                '<!-- Modal content-->' +
                '<div class="modal-content">' +
                  '<div class="modal-header">' +
                    '<button type="button" class="close" data-dismiss="modal">&times;</button>' +
                    '<h4 class="modal-title">Choose a story</h4>' +
                  '</div>' +
                  '<div class="modal-body modal-body-content">' +
                  '</div>' +
                  '<div class="modal-footer">' +
                    '<button type="button" class="btn btn-default btn-close" data-dismiss="modal">Close</button>' +
                  '</div>' +
                '</div>' +

              '</div>' +

            '</div>';

});