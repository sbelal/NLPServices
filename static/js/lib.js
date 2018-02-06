$(function(){
	$('#btnSubmit').click(function(){
		
		text = $('#inputName').val()
		
		$.ajax({
			url: '/api/v1.0/sentimentscore',
			data: JSON.stringify({'text':text}),
			contentType : 'application/json',
			type: 'POST',
			success: function(response){
                score =parseFloat(response.score)
                
                answerText = score + " -- " + "NEGATIVE"
                if(score > 0.5)
                {
                    answerText = score + " -- " + "POSITIVE"
                }

                $('#responseText').text(answerText);
                console.log(response);
			},
			error: function(error){
                $('#responseText').text("ERROR");
				console.log(error);
			}
		});
	});
});

function test()
{
	return ""
}