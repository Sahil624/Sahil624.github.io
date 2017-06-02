$('document').ready(function(){
    console.log('Ready');
    var pf = $('#portfolio').offset().top;
        var about = $("#about").offset().top;
        var contact = $('#contact').offset().top;
    console.log(pf,about,contact);

    $(window).scroll(function(){
        var scrolled = $(window).scrollTop();
        var height = $(window).height()
       // var scbtm = scrolled + + $(window).height();
        if(scrolled >= 50){
            $('nav').addClass('shrink');
        }
        else{
            $('nav').removeClass('shrink')
        }



        if(scrolled>=pf-100 && scrolled <about-50){
            $('.pf-btn').addClass('active');
            $('.ab-btn').removeClass('active');
            $('.cn-btn').removeClass('active');
        }

        else if(scrolled >= about-50 && scrolled < contact-100){
            $('.pf-btn').removeClass('active');
            $('.ab-btn').addClass('active');
            $('.cn-btn').removeClass('active');
        }

        else if(scrolled >= contact-100){
            $('.pf-btn').removeClass('active');
            $('.ab-btn').removeClass('active');
            $('.cn-btn').addClass('active');
        }

        else{
            $('.pf-btn').removeClass('active');
            $('.ab-btn').removeClass('active');
            $('.cn-btn').removeClass('active');
        }
        });

    $('#img').click(function(){
        console.log('Clicked');
    })

    $('.img-responsive').hover(function(){
        $('.caption-content').show(100);
        $(this).css({opacity:.5+'px'});
        console.log('Show');
    },function(){
        $('.caption-content').hide(500);
        $(this).css({opacity:1+'px'});
        console.log('Hide');
    })
})
