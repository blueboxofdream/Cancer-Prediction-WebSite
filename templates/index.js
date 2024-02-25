function validateSignIn() {
    let email = document.getElementById('email').value;
    let password = document.getElementById('password').value;

    if (email === "") {
        alert("Please enter your email");
        return false;
    }
    if (password === "") {
        alert("Please enter your password.");
        return false;
    }
    fetch('your-signin-api-endpoint', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Sign Up Success:', data);
    })
    .catch(error => {
        console.error('Sign Up Error:', error);
    });
    return true; 
}

function validateSignUp() {
    let name = document.getElementById('username').value;
    let email = document.getElementById('email').value;
    let password = document.getElementById('password').value;

    if (name === "") {
        alert("Please enter your name.");
        return false;
    }

    if (email === "") {
        alert("Please enter your email.");
        return false;
    }

    if (password === "") {
        alert("Please enter your password.");
        return false;
    }
    fetch('your-signup-api-endpoint', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, email, password }),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Sign Up Success:', data);
    })
    .catch(error => {
        console.error('Sign Up Error:', error);
    });
    return true;
}