async function SendChatBot() {
 var inputQuestion = document.getElementById("inputQuestion").value;

    // Send the question to the backend
    var response = await fetch("http://localhost:5000/API", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ message: inputQuestion })
    });
    document.getElementById("inputQuestion").value = ""; // Clear the input field after sending
    var IA_response = await response.json()
    document.getElementById("ChatContainer").innerHTML +=  "<br>" + IA_response.reponse

}