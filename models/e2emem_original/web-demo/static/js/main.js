requirejs.config({
    baseUrl: "js",
    paths: {
        "jquery": "lib/jquery/jquery",
        "underscore": "lib/underscore/underscore",
        "backbone": "lib/backbone/backbone",
        "jquery-ui": "lib/jquery-ui.min",
        "bootstrap": "lib/bootstrap.min"
    },
    shim: {
        "underscore": {
            deps: [],
            exports: "_"
        },
        "backbone": {
            deps: ["jquery", "underscore"],
            exports: "Backbone"
        },
        "bootstrap": {
            deps: ["jquery", "underscore"]
        }
    }
});

define(["app", "bootstrap"], function(App) {
});
