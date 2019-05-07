//app.js
// Define a new module for our app. The array holds names of dependencies if any.
var app = angular.module("searchApp", []);
var topics = {};
// Create the search filter
app.filter('searchFor', function(){
    // The first parameter
    // is the data that is to be filtered, and the second is an
    // argument that may be passed with a colon (searchFor:searchString)
    return function(dataArr, searchString){

        if(!searchString){
            return [];
        }

        var result;
        searchString = searchString.toLowerCase();
        result = dataArr.map(function(curVal) {
            for(var i=0; i<curVal.fc_content.length; i++){
                 if(curVal.fc_content[i].indexOf(searchString) !== -1){
                     return curVal;
                 }
            }
        });
        // Using the forEach helper method to loop through the array
        /*angular.forEach(dataArr, function(item){

            for(var i = 0, len = item.fc_content.length; i < len; i++){
                if(item.fc_content[i].toLowerCase().indexOf(searchString) !== -1){
                   console.log(item.fc_content[i]);
                    result.push(item);
                }
            }

        });*/
       console.log(result);
        return result;
    };
});
app.filter('removeBlankItems', function() {
    return function(array) {
        return array.filter(function (o) {
            return o !== null;
        });}});

// The controller
function searchController($scope,$http){

    $http.get('https://raw.githubusercontent.com/sujithksam92/SearchEngineProject/master/testdata.json')
            .then(function(res){
            $scope.dataDump = res.data;
        });
    $http.get('https://raw.githubusercontent.com/sujithksam92/SearchEngineProject/master/df_dominant_topics.json')
        .then(function(res){
            topics = res.data;
        });
}