import {createRouter, createWebHistory} from 'vue-router'
import inputTest from "../views/inputTest.vue";

const routerHistory = createWebHistory();
const router = createRouter({
    history: routerHistory,
    routes: [
        {
            path: '/',
            component:inputTest
        },
        {
            path: '/home',
            component: () => import('../views/home.vue')
        }
    ]
})

export default router