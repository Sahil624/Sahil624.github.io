$('.burger').click(function(){
    $('.ham').toggleClass('\glyphicon-menu-hamburger');
    $('.ham').toggleClass('glyphicon-chevron-left');
    if($('.links').css('display') == 'none'){
        $("nav").css("background-color", "rgba(150,125,255,.9)");
    //$('nav').css('bacground-color','rgba(255,255,255,.7)');
    $('.links').slideDown('500');
    }
    else{
       $('.links').slideUp('500'); 
        $("nav").css("background-color", "transparent");
    }
})

// nAv