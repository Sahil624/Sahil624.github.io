$('.burger').click(function(){
    $('.ham').toggleClass('\glyphicon-menu-hamburger');
    $('.ham').toggleClass('glyphicon-chevron-left');
    if($('.links').css('display') == 'none'){
    $('.links').slideDown('500');
    }
    else{
       $('.links').slideUp('500'); 
    }
})