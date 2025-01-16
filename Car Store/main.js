var productlist = [];

function add(x) {
    productlist.push(x);
    localStorage.setItem("car", JSON.stringify(productlist));
    disp(x);
}

function disp(y) {
    productlist = JSON.parse(localStorage.getItem("car"));
    var carto = "";
    for (let i = 0; i < productlist.length; i++) {
        carto += `<h1>${y.carName}</h1>`;
    }
    document.getElementById("cart").innerHTML = carto;
}

const warehouse = [
    { carName: "Alfa Romeo Tonale", quantity: 2 },
    { carName: "Alfa Romeo Giulia", quantity: 2 }, // 1
    { carName: "Alfa Romeo Stelvio", quantity: 1 },
    { carName: "Audi A6", quantity: 0 },
    { carName: "Audi Q3", quantity: 2 }, // 2
    { carName: "Audi Q8", quantity: 4 },
    { carName: "BMW 3 Series", quantity: 1 },
    { carName: "BMW 5 Series", quantity: 2 }, // 3
    { carName: "BMW 7 Series", quantity: 2 },
    { carName: "BMW M8", quantity: 1 },
    { carName: "BMW X6", quantity: 1 },
    { carName: "BMW X7", quantity: 2 },
    { carName: "Ford Fusion", quantity: 1 },
    { carName: "Ford Ecosport", quantity: 1 }, // 5
    { carName: "Ford Mustang", quantity: 1 },
    { carName: "Mercedes CLA", quantity: 0 },
    { carName: "Mercedes A Class", quantity: 1 },
    { carName: "Mercedes B Class", quantity: 1 },
    { carName: "Mercedes C Class", quantity: 1 }, // 6
    { carName: "Mercedes E Class", quantity: 2 },
    { carName: "Mercedes S Class", quantity: 2 },
    { carName: "Kia Cerato", quantity: 2 }, // 25
    { carName: "Kia Sportage", quantity: 0 }, // 26
    { carName: "Kia Picanto", quantity: 2 }, // 27
    { carName: "Nissan Pathfinder", quantity: 1 }, // 28
    { carName: "Nissan Titan", quantity: 2 }, // 29
    { carName: "Nissan GTR", quantity: 2 }, // 30
    { carName: "Dodge Charger", quantity: 1 }, // 25
    { carName: "Dodge Durango", quantity: 2 }, // 26
    { carName: "Dodge Challenger", quantity: 2 }, // 27
    { carName: "Chevorlet Camaro", quantity: 0 },
    { carName: "Chevorlet Captiva", quantity: 2 },
    { carName: "Chevorlet Cruze", quantity: 2 },
    { carName: "Lamborghini Hurcan", quantity: 4 },
    { carName: "Lamborghini Urus", quantity: 0 },
    { carName: "Lamborghini Aventador", quantity: 2 },
    { carName: "Toyota Land Cruiser", quantity: 1 },
    { carName: "Toyota RAV4", quantity: 0 },
    { carName: "Toyota Corolla", quantity: 4 },
    { carName: "Volvo XC40", quantity: 0 },
    { carName: "Volvo XC60", quantity: 2 },
    { carName: "Volvo XC90", quantity: 1 },
];

function m(element) {
    const i = parseInt(element.value);
    const resultContainer = document.getElementById("transaction-result");
    const car = warehouse[i];

    if (!car) {
        alert("Invalid car selection.");
        return;
    }

    let cart = "";
    if (car.quantity > 0) {
        car.quantity -= 1;

        cart = `<h1>${car.carName} has been purchased.<br> Remaining stock: ${car.quantity}</h1> 
        <br>
        <h3>If you want this car you should create an account</h3>
        <a href="../cart.html" class="cart-btn">go to cart</a>`;
        resultContainer.innerHTML = cart;
        add(car);
    } else {
        resultContainer.innerHTML = `<h1>Sorry, ${car.carName} is out of stock.</h1>`;
    }

    document.getElementById("car").innerHTML = cart;
}

//-------------------------------------------------------Mohamed hossam part start here-----------------------------------------------------------------------------------------//
/*-------------------------------------------------------Search--------------------------------------------------------*/
function searchBrand() {
    var keyword = document.getElementById("searchInput").value.toLowerCase();

    var Brands = {
        "bmw": "https://autozone-cargallery.github.io/BMW.html",
        "audi": "https://autozone-cargallery.github.io/Audi/Audi.html",
        "mercedes": "https://autozone-cargallery.github.io/Mercedes.html",
        "alfa romeo": "https://autozone-cargallery.github.io/Alfa-Romeo/Alfa.html",
        "ford": "https://autozone-cargallery.github.io/Ford.html",
        "kia": "https://autozone-cargallery.github.io/Desoky/Kia.html",
        "nissan": "https://autozone-cargallery.github.io/Nissan%20cars%20shop/Nissan%20cars.html",
        "dodge": "https://autozone-cargallery.github.io/Dodge.html",
        "chevrolet": "https://autozone-cargallery.github.io/Cars/Chevy%20cars/Chevrolet.html",
        "lamborghini": "https://autozone-cargallery.github.io/Cars/Lamborgini%20Cars/Lamborghini.html",
        "toyota": "https://autozone-cargallery.github.io/Toyota/Touota.html",
        "volvo": "https://autozone-cargallery.github.io/Volvo.html",
        "bmw 3 series": "https://autozone-cargallery.github.io/BMW/BMW%203%20Series.html" ,
        "bmw 5 series": "https://autozone-cargallery.github.io/BMW/BMW%205%20Series.html" ,
        "bmw 7 series": "https://autozone-cargallery.github.io/BMW/BMW%207%20Series.html" ,
        "bmw m8": "https://autozone-cargallery.github.io/BMW/BMW%20M8.html" ,
        "bmw x6": "https://autozone-cargallery.github.io/BMW/BMW%20X6.html" ,
        "bmw x7": "https://autozone-cargallery.github.io/BMW/BMW%20X7.html" ,
        "mercedes cla": "https://autozone-cargallery.github.io/Mercedes/Mercedes%20CLA.html" ,
        "mercedes a class": "https://autozone-cargallery.github.io/Mercedes/Mercedes%20A%20Class.html" ,
        "mercedes b class": "https://autozone-cargallery.github.io/Mercedes/Mercedes%20B%20Class.html" ,
        "mercedes c class": "https://autozone-cargallery.github.io/Mercedes/Mercedes%20C%20Class.html" ,
        "mercedes e class": "https://autozone-cargallery.github.io/Mercedes/Mercedes%20E%20Class.html" ,
        "mercedes s class": "https://autozone-cargallery.github.io/Mercedes/Merccedes%20S%20Class.html" ,
    };
    if (Brands.hasOwnProperty(keyword)) {
        window.location.href = Brands[keyword];
    } else {
        alert("Check spelling or else this brand doesn't exist on website....");
    }
}
/*-------------------------------------------------------Search--------------------------------------------------------*/
/*-------------------------------------------------------Message--------------------------------------------------------*/
//let addToCartButtons = document.querySelectorAll(".add-to-cart");

//addToCartButtons.forEach(function (button) {
//    button.addEventListener("click", function () {
//        var carName = this.getAttribute("onclick");
//        addToCart(carName);
//    });
//});

//function addToCart(carName) {
//    alert("" + carName);
//}
/*-------------------------------------------------------Message--------------------------------------------------------*/
/*-------------------------------------------------------Photos Of Cars--------------------------------------------------------*/
document.addEventListener("DOMContentLoaded", function () {
    const imageSets = [
        ["../Cars/BMW/3/1.png", "../Cars/BMW/3/2.png", "../Cars/BMW/3/3.png", "../Cars/BMW/3/4.jpg"],
        ["../Cars/BMW/5/1.jpg", "../Cars/BMW/5/2.jpg", "../Cars/BMW/5/3.jpg", "../Cars/BMW/5/4.jpg", "../Cars/BMW/5/5.jpg", "../Cars/BMW/5/6.jpg", "../Cars/BMW/5/7.jpg", "../Cars/BMW/5/8.jpg"],
        ["../Cars/BMW/7/1.jpg", "../Cars/BMW/7/2.jpg", "../Cars/BMW/7/3.jpg", "../Cars/BMW/7/4.jpg", "../Cars/BMW/7/5.jpg", "../Cars/BMW/7/6.jpg", "../Cars/BMW/7/7.jpg", "../Cars/BMW/7/8.jpg", "../Cars/BMW/7/9.jpg"],
        ["../Cars/BMW/BMW M8 Gran Coupe/1.jpg", "../Cars/BMW/BMW M8 Gran Coupe/2.jpg", "../Cars/BMW/BMW M8 Gran Coupe/3.jpg", "../Cars/BMW/BMW M8 Gran Coupe/4.jpg", "../Cars/BMW/BMW M8 Gran Coupe/5.jpg", "../Cars/BMW/BMW M8 Gran Coupe/6.jpg", "../Cars/BMW/BMW M8 Gran Coupe/7.jpg", "../Cars/BMW/BMW M8 Gran Coupe/8.jpg", "../Cars/BMW/BMW M8 Gran Coupe/9.jpg", "../Cars/BMW/BMW M8 Gran Coupe/10.jpg"],
        ["../Cars/BMW/X6/2024-bmw-x6-110-1675791921.jpg", "../Cars/BMW/X6/2024-bmw-x6-113-1675791922.jpg", "../Cars/BMW/X6/P90495591_highRes_the-new-bmw-x6-m-com.jpg"],
        ["../Cars/BMW/X7/2023-bmw-x7-xdrive-40i144-641c5b4169eb2.jpg", "../Cars/BMW/X7/2023-bmw-x7-xdrive-40i147-641c5b438a1cc.jpg", "../Cars/BMW/X7/2023-bmw-x7-xdrive-40i128-641c5b39edaad.jpg", "../Cars/BMW/X7/2023-bmw-x7-xdrive-40i132-641c5b3c33cbc.jpg", "../Cars/BMW/X7/2023-bmw-x7-xdrive-40i101-641c5b2d70bc9.jpg", "../Cars/BMW/X7/2023-bmw-x7-xdrive-40i115-641c5b31db5bb.jpg"],
        ["../Cars/Mercedes/CLA/1.jpg", "../Cars/Mercedes/CLA/2.webp", "../Cars/Mercedes/CLA/3.jpg", "../Cars/Mercedes/CLA/4.jpg"],
        ["../Cars/Mercedes/A_Class/3.jpg", "../Cars/Mercedes/A_Class/4.jpg", "../Cars/Mercedes/A_Class/5.jpg", "../Cars/Mercedes/A_Class/6.jpg", "../Cars/Mercedes/A_Class/1.jpg", "../Cars/Mercedes/A_Class/2.jpg", "../Cars/Mercedes/A_Class/7.jpg", "../Cars/Mercedes/A_Class/8.jpg"],
        ["../Cars/Mercedes/B_Class/640px-Mercedes-Benz_B-Class.jpg", "../Cars/Mercedes/B_Class/2.jpg", "../Cars/Mercedes/B_Class/Mercedes-Benz_B-Class_at_the_Frankfurt_Motor_Show_IAA_2011_(6147726918).jpg"],
        ["../Cars/Mercedes/C_Class/1.jpg", "../Cars/Mercedes/C_Class/2.jpg", "../Cars/Mercedes/C_Class/3.jpg", "../Cars/Mercedes/C_Class/4.jpg", "../Cars/Mercedes/C_Class/5.jpg", "../Cars/Mercedes/C_Class/6.jpg", "../Cars/Mercedes/C_Class/7.jpg", "../Cars/Mercedes/C_Class/8.jpg"],
        ["../Cars/Mercedes/E_Class/4.jpg", "../Cars/Mercedes/E_Class/5.jpg", "../Cars/Mercedes/E_Class/6.jpg", "../Cars/Mercedes/E_Class/7.jpg", "../Cars/Mercedes/E_Class/1.jpg", "../Cars/Mercedes/E_Class/2.jpg", "../Cars/Mercedes/E_Class/3.jpg", "../Cars/Mercedes/E_Class/8.jpg", "../Cars/Mercedes/E_Class/9.jpg"],
        ["../Cars/Mercedes/S_Class/2022-mercedes-benz-s500-4matic-101-1642184014.jpg", "../Cars/Mercedes/S_Class/2022-mercedes-benz-s500-4matic-102-1642184016.jpg", "../Cars/Mercedes/S_Class/2022-mercedes-benz-s500-4matic-103-1642184013.jpg", "../Cars/Mercedes/S_Class/2022-mercedes-benz-s500-4matic-114-1642184019.jpg", "../Cars/Mercedes/S_Class/2022-mercedes-benz-s500-4matic-124-1642184029.jpg"],
        ["../Cars/Alfa/Tonale/2024-alfa-romeo-tonale-eawd-103-6525bc16c793d.jpg", "../Cars/Alfa/Tonale/2024-alfa-romeo-tonale-eawd-237-6525bc17be9bb.jpg","../Cars/Alfa/Tonale/2024-alfa-romeo-tonale-eawd-316-6525bd7902872.jpg","../Cars/Alfa/Tonale/2024-alfa-romeo-tonale-eawd-496-6525bc1f91c71.jpg","../Cars/Alfa/Tonale/2024-alfa-romeo-tonale-eawd-511-6525bc219f129.jpg","../Cars/Alfa/Tonale/2024-alfa-romeo-tonale-eawd-569-6525bc29c2a98.jpg"],
        ["../Cars/Alfa/Giulia/2023-alfa-romeo-giulia-101-6400fe6ad4891.jpeg","../Cars/Alfa/Giulia/2023-alfa-romeo-giulia-102-6400fe6b0bb68.jpeg","../Cars/Alfa/Giulia/2023-alfa-romeo-giulia-103-6400fe6b10c39.jpeg","../Cars/Alfa/Giulia/2023-alfa-romeo-giulia-107-6400fe6bccb6a.jpeg","../Cars/Alfa/Giulia/2023-alfa-romeo-giulia-110-6400fe6ae4216.jpeg"],
        ["../Cars/Alfa/Stelvio/2024-alfa-romeo-stelvio-105-640215298fa2f.jpeg","../Cars/Alfa/Stelvio/2024-alfa-romeo-stelvio-106-640215298b274.jpeg","../Cars/Alfa/Stelvio/2024-alfa-romeo-stelvio-107-64021529f3b4c.jpeg","../Cars/Alfa/Stelvio/2024-alfa-romeo-stelvio-114-6402152ad8885.jpeg","../Cars/Alfa/Stelvio/2024-alfa-romeo-stelvio-115-6402152ada34e.jpeg"],
        ["../Cars/Audi/A6/2024-audi-a6-101-64761a190970d.jpg","../Cars/Audi/A6/2024-audi-a6-102-64761a1910bef.jpg","../Cars/Audi/A6/2024-audi-a6-105-64761a1ab6eb4.jpg","/Cars/Audi/A6/2024-audi-s6-102-64761a1bb689c.jpg"],
        ["../Cars/Audi/Q3/2022-audi-q3-premium-45-119-64b19da0646ad.jpg","../Cars/Audi/Q3/2022-audi-q3-premium-45-118-64b19da02e90e.jpg","../Cars/Audi/Q3/2022-audi-q3-premium-45-116-64b19da017a6a.jpg","../Cars/Audi/Q3/2022-audi-q3-premium-45-101-64b19d9d9cfbd.jpg","../Cars/Audi/Q3/2022-audi-q3-premium-45-105-64b19d9dec649.jpg"],
        ["../Cars/Audi/Q8/2024-audi-q8-exterior-static-107-64f0ba5434e59.jpg","../Cars/Audi/Q8/2024-audi-q8-exterior-static-106-64f0ba52ee821.jpg","../Cars/Audi/Q8/2024-audi-q8-exterior-static-101-64f0ba536838c.jpg","../Cars/Audi/Q8/2024-audi-q8-interior-101-64f0b9ce6969a.jpg","../Cars/Audi/Q8/2024-audi-q8-interior-102-64f0b9cd53b02.jpg","../Cars/Audi/Q8/2024-audi-q8-interior-103-64f0b9cd749a7.jpg"],
        ["../Cars/Ford/2020 Ford Fusion/2020-ford-fusion-mmp-1-1568742907.jpeg","../Cars/Ford/2020 Ford Fusion/2020-ford-fusion-mmp-2-1568743032.jpeg","../Cars/Ford/2020 Ford Fusion/2020-ford-fusion-mmp-3-1568743081.jpeg"],
        ["../Cars/Ford/2022 Ford EcoSport/2018-ford-ecosport-2-0l-102-1527708025.jpg","../Cars/Ford/2022 Ford EcoSport/2018-ford-ecosport-2-0l-103-1527708026.jpg","../Cars/Ford/2022 Ford EcoSport/2018-ford-ecosport-2-0l-109-1527708027.jpg","../Cars/Ford/2022 Ford EcoSport/2018-ford-ecosport-2-0l-115-1527708033.jpg","../Cars/Ford/2022 Ford EcoSport/2018-ford-ecosport-2-0l-122-1527708031.jpg","../Cars/Ford/2022 Ford EcoSport/2018-ford-ecosport-2-0l-125-1527708032.jpg","../Cars/Ford/2022 Ford EcoSport/2018-ford-ecosport-2-0l-141-1527708038.jpg"],
        ["../Cars/Ford/Mustang/ford-mustang-shelby-gt500-2022-651234.jpg","../Cars/Ford/Mustang/ford-mustang-shelby-gt500-2022-651226.jpg","../Cars/Ford/Mustang/gettyimages-475728306-612x612.jpg"],
        ["../Cars/Nissan/Pathfinder/01-2024-nissan-pathfinder-angular-front-jms.jpeg","../Cars/Nissan/Pathfinder/2022-nissan-pathfinder-platinum-awd-270-1637092224.jpg"],
        ["../Cars/Nissan/Titan/wp2393298-nissan-titan-wallpapers.jpg","../Cars/Nissan/Titan/wp2393332-nissan-titan-wallpapers.jpg"],
        ["../Cars/Nissan/GT-R/jacob-s-3VVp9aOUJEY-unsplash.jpg","../Cars/Nissan/GT-R/martin-katler-4A0Z_FxXpVk-unsplash.jpg"],
        ["../Cars/KIA/Cerato/kia_k3_bd_pe_22my_gallery_exterior_01.jpg","../Cars/KIA/Cerato/kia_k3_bd_pe_22my_gallery_exterior_03.jpg","../Cars/KIA/Cerato/kia_k3_bd_pe_22my_gallery_interior_01.jpg"],
        ["../Cars/KIA/Sportage/kia-nq5-22my-thum-exterior-01.jpg","../Cars/KIA/Sportage/kia-nq5-22my-thum-exterior-02.jpg"],
        ["../Cars/KIA/Picanto/picanto-24my-thum-exterior-05.jpg","../Cars/KIA/Picanto/picanto-24my-thum-exterior-06.jpg","../Cars/KIA/Picanto/picanto-24my-thum-interior-11.jpg"],
        ["../Cars/Dodge/Charger/pexels-kelcey-shotit-696990889-19244520.jpg","../Cars/Dodge/Charger/pexels-04iraq-1272398525-23952894.jpg"],
        ["../Cars/Dodge/Durango/pexels-alba-cars-139933177-11592843.jpg","../Cars/Dodge/Durango/pexels-messina-photo-61078602-8042432.jpg"],
        ["../Cars/Dodge/Challenger/pexels-jakewymoore-12330341.jpg","../Cars/Dodge/Challenger/pexels-mike-noga-1916934-3608542.jpg"],
        ["../Cars/Chevy/Camaro/chevycamaro.jpg","../Cars/Chevy/Camaro/pexels-04iraq-1272398525-29098284.jpg"],
        ["../Cars/Chevy/Cativa/Chevycapvita.jpg","../Cars/Chevy/Cativa/images.jpeg"],
        ["../Cars/Chevy/Cruze/Chevycruze.jpg","../Cars/Chevy/Cruze/download.jpeg"],
        ["../Cars/Lamborgini/Hurcan/LamborginiHurcan.jpg","../Cars/Lamborgini/Hurcan/michael-lock-NfbUoOFUuco-unsplash.jpg","../Cars/Lamborgini/Hurcan/viktor-theo-jh8T0oqzcZo-unsplash.jpg"],
        ["../Cars/Lamborgini/Urus/Lamborghini_Urus_19.09.20_JM_(2)_(cropped).jpg","../Cars/Lamborgini/Urus/LamborginiUrus.jpg","../Cars/Lamborgini/Urus/pexels-katetrysh-27504024.jpg"],
        ["../Cars/Lamborgini/Aventador/adrian-newell-6TcjU3i8d8s-unsplash.jpg","../Cars/Lamborgini/Aventador/LamborginiAventador.jpg","../Cars/Lamborgini/Aventador/marc-kleen-8hU6vtwY8l8-unsplash.jpg"],
        ["../Cars/Toyota/Cruizer/depositphotos_514058360-stock-photo-russia-izhevsk-september-2021-toyota.jpg","../Cars/Toyota/Cruizer/depositphotos_633682612-stock-photo-pasay-apr-toyota-land-cruiser.jpg"],
        ["../Cars/Toyota/RAV4/depositphotos_437525596-stock-photo-ufa-russia-15-november-2020.jpg","../Cars/Toyota/RAV4/depositphotos_250845458-stock-photo-toyota-rxv4-2019-interior.jpg","../Cars/Toyota/RAV4/depositphotos_614682802-stock-photo-toyota-rav4-hybrid-car-showcased.jpg"],
        ["../Cars/Toyota/Corolla/istockphoto-1412133515-612x612.jpg","../Cars/Toyota/Corolla/istockphoto-1442768053-612x612.jpg","../Cars/Toyota/Corolla/istockphoto-1490891378-612x612.jpg"],
        ["../Cars/Volvo/XC40/gettyimages-1175886362-612x612.jpg","../Cars/Volvo/XC40/gettyimages-1199188964-612x612.jpg","../Cars/Volvo/XC40/gettyimages-1199189059-612x612.jpg"],
        ["../Cars/Volvo/XC60/gettyimages-1199188661-612x612.jpg","../Cars/Volvo/XC60/gettyimages-1327543751-612x612.jpg","../Cars/Volvo/XC60/gettyimages-649701514-612x612.jpg"],
        ["../Cars/Volvo/XC90/istockphoto-1318131076-612x612.jpg","../Cars/Volvo/XC90/istockphoto-1499274436-612x612.jpg","../Cars/Volvo/XC90/istockphoto-1891879143-612x612.jpg"],
        
    ];

    if (document.getElementById("BMW")) {
        for (let i = 0; i < imageSets.length; i++) {
            initializeViewer(i, imageSets[i]);
        }
    } else {
        for (let i = 6; i < imageSets.length; i++) {
            initializeViewer(i, imageSets[i]);
        }
    }

    function initializeViewer(index, images) {
        const imageViewer = document.getElementById(`image-viewer-${index + 1}`);
        if (!imageViewer) return;

        const backwardButton = imageViewer.querySelector(".backward-btn");
        const forwardButton = imageViewer.querySelector(".forward-btn");
        const imageElement = imageViewer.querySelector(".current-image");

        let currentIndex = 0;

        function showImage(i) {
            currentIndex = (i % images.length + images.length) % images.length;
            imageElement.src = images[currentIndex];
        }

        backwardButton.addEventListener("click", function () {
            showImage(--currentIndex);
        });

        forwardButton.addEventListener("click", function () {
            showImage(++currentIndex);
        });

        // Show the initial image
        showImage(currentIndex);
    }
});
/*-------------------------------------------------------Photos Of Cars--------------------------------------------------------*/
/*---------------------------------------Feedback-----------------------------------------------------------*/
function openFeedbackBox() {
    document.getElementById('feedbackBox').style.display = 'block';
}

function closeFeedbackBox() {
    document.getElementById('feedbackBox').style.display = 'none';
}

function submitFeedback() {
    const feedbackText = document.getElementById('feedbackText').value;
    if (feedbackText.trim() === "") {
        alert("Please enter your feedback before submitting.");
    } else {
        alert("Thank you for your feedback!");
        closeFeedbackBox();
    }
}
/*---------------------------------------Feedback-----------------------------------------------------------*/

/*------------------------------------------Login-----------------------------------------------------*/
document.getElementById("signup-form").addEventListener("submit", function(event) {
    event.preventDefault();
});
/*------------------------------------------Login-----------------------------------------------------*/