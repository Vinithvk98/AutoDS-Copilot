/* Subtle, professional motion: scroll reveal + stat count-up.
   Honours prefers-reduced-motion and degrades to no animation. */
(function(){
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // sticky-nav hairline
  var nav = document.getElementById("nav");
  if(nav) window.addEventListener("scroll", function(){
    nav.classList.toggle("scrolled", window.scrollY > 8);
  });

  var reveals = [].slice.call(document.querySelectorAll(".reveal"));
  function countUp(el){
    var target = parseFloat(el.getAttribute("data-count")); if(isNaN(target)) return;
    var suffix = /\+/.test(el.textContent) ? "+" : "";
    if(reduce){ el.textContent = target + suffix; return; }
    var start = null, dur = 900;
    function step(ts){ if(!start) start = ts; var p = Math.min((ts-start)/dur, 1);
      var val = Math.round(target * (0.5 - Math.cos(Math.PI*p)/2)); // ease in-out
      el.textContent = val + suffix; if(p < 1) requestAnimationFrame(step); }
    el.textContent = "0" + suffix; requestAnimationFrame(step);
  }

  if(reduce || !("IntersectionObserver" in window)){
    reveals.forEach(function(el){ el.classList.add("in"); });
    document.querySelectorAll("[data-count]").forEach(countUp);
    return;
  }

  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting){
        e.target.classList.add("in");
        e.target.querySelectorAll("[data-count]").forEach(countUp);
        io.unobserve(e.target);
      }
    });
  }, {threshold: 0.12, rootMargin: "0px 0px -40px 0px"});
  reveals.forEach(function(el){ io.observe(el); });

  // count-up for stat numbers that are not inside a .reveal
  var solo = [].slice.call(document.querySelectorAll("[data-count]")).filter(function(el){
    return !el.closest(".reveal"); });
  if(solo.length){
    var io2 = new IntersectionObserver(function(entries){ entries.forEach(function(e){
      if(e.isIntersecting){ countUp(e.target); io2.unobserve(e.target); } }); }, {threshold:0.5});
    solo.forEach(function(el){ io2.observe(el); });
  }
})();
